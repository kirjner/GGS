from typing import List, Optional, Tuple
import hydra
from biotite.sequence.io import fasta
from polyleven import levenshtein
import numpy as np
import torch
import pyrootutils
import logging
import os
from omegaconf import DictConfig
import pandas as pd
from ggs import utils
from ggs.models.predictors import BaseCNN
from omegaconf import OmegaConf
from ggs.data.utils.tokenize import Encoder
import glob
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

log = utils.get_pylogger(__name__)
to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()
alphabet = "ARNDCQEGHILKMFPSTWYV"

def diversity(seqs):
    num_seqs = len(seqs)
    total_dist = 0
    for i in range(num_seqs):
        for j in range(num_seqs):
            x = seqs[i]
            y = seqs[j]
            if x == y:
                continue
            total_dist += levenshtein(x, y)
    return total_dist / (num_seqs*(num_seqs-1))

def _read_fasta(fasta_path):
    fasta_seqs = fasta.FastaFile.read(fasta_path)
    seq_to_fitness = {}
    process_header = lambda x: float(x.split('_')[-1].split('=')[1])
    for x,y in fasta_seqs.items():
        seq_to_fitness[y] = process_header(x)
    return seq_to_fitness

class EvalRunner:
    def __init__(self, runner_cfg):
        self._cfg = runner_cfg
        self._log = logging.getLogger(__name__)
        self.predictor_tokenizer = Encoder()
        gt_csv = pd.read_csv(self._cfg.gt_csv)
        gt_csv = gt_csv[gt_csv.augmented == 0]
        oracle_dir = self._cfg.oracle_dir
        self.use_normalization = self._cfg.use_normalization
        # Read in known sequences and their fitnesses
        self._max_known_score = np.max(gt_csv.target)
        self._min_known_score = np.min(gt_csv.target)
        self.normalize = lambda x: to_np((x - self._min_known_score) / (self._max_known_score - self._min_known_score)).item()
        self._log.info(f'Read in {len(gt_csv)} ground truth sequences.')
        self._log.info(f'Maximum known score {self._max_known_score}.')
        self._log.info(f'Minimum known score {self._min_known_score}.')

        # Read in base pool used to generate sequences.
        base_pool_seqs = pd.read_csv(self._cfg.base_pool_path)
        self._base_pool_seqs = base_pool_seqs.sequence.tolist()
        log.info(f'Read in {len(base_pool_seqs)} base pool sequences.')
        self._log.info(f'Maximum base score {base_pool_seqs.target.max()}.')
        self._log.info(f'Minimum base score {base_pool_seqs.target.min()}.')
        self.device = torch.device('cuda') #requires GPU
        self._log.info(f'Running on GPU: {self.device}.')
        oracle_path = os.path.join(oracle_dir, 'cnn_oracle.ckpt')
        oracle_state_dict = torch.load(oracle_path, map_location=self.device)
        cfg_path = os.path.join(oracle_dir, 'config.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)

        self._cnn_oracle = BaseCNN(**ckpt_cfg.model.predictor) #oracle has same architecture as predictor
        self._cnn_oracle.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in oracle_state_dict['state_dict'].items()})
        self._cnn_oracle = self._cnn_oracle.to(self.device)
        self._cnn_oracle.eval()
        if self._cfg.predictor_dir is not None:
            predictor_path = os.path.join(self._cfg.predictor_dir, 'last.ckpt')
            predictor_state_dict = torch.load(predictor_path, map_location=self.device)
            self._predictor = BaseCNN(**ckpt_cfg.model.predictor) #oracle has same architecture as predictor
            self._predictor.load_state_dict(
                {k.replace('predictor.', ''): v for k,v in predictor_state_dict['state_dict'].items()})
            self._predictor = self._predictor.to(self.device)
        self.run_oracle = self._run_cnn_oracle
        self.run_predictor = self._run_predictor if self._cfg.predictor_dir is not None else None


    def novelty(self, sampled_seqs):
        # sampled_seqs: top k
        # existing_seqs: range dataset
        all_novelty = []
        for src in tqdm(sampled_seqs):  
            min_dist = 1e9
            for known in self._base_pool_seqs:
                dist = levenshtein(src, known)
                if dist < min_dist:
                    min_dist = dist
            all_novelty.append(min_dist)
        return all_novelty

    def tokenize(self, seqs):
        return self.predictor_tokenizer.encode(seqs).to(self.device)

    def _run_cnn_oracle(self, seqs):
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self._cfg.batch_size, 0)
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self._cnn_oracle(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0)

    def _run_predictor(self, seqs):
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self._cfg.batch_size, 0)
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self._predictor(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0)
    
    def evaluate_sequences(self, topk_seqs, use_oracle = True):
        topk_seqs = list(set(topk_seqs))
        num_unique_seqs = len(topk_seqs)
        topk_scores = self.run_oracle(topk_seqs) if use_oracle else self.run_predictor(topk_seqs)
        normalized_scores = [self.normalize(x) for x in topk_scores]
        seq_novelty = self.novelty(topk_seqs)
        results_df = pd.DataFrame({
            'sequence': topk_seqs,
            'oracle_score': to_list(topk_scores),
            'normalized_score': normalized_scores,
            'novelty': seq_novelty,
        })  if use_oracle else pd.DataFrame({
            'sequence': topk_seqs,
            'predictor_score': to_list(topk_scores),
            'normalized_score': normalized_scores,
            'novelty': seq_novelty,
        })

        if num_unique_seqs == 1:
            seq_diversity = 0
        else:
            seq_diversity = diversity(topk_seqs)
               
        metrics_scores = normalized_scores if self.use_normalization else topk_scores.detach().cpu().numpy()
        metrics_df = pd.DataFrame({
            'num_unique': [num_unique_seqs],
            'mean_fitness': [np.mean(metrics_scores)],
            'mean_fitness': [np.mean(metrics_scores)],
            'median_fitness': [np.median(metrics_scores)],
            'std_fitness': [np.std(metrics_scores)],
            'max_fitness': [np.max(metrics_scores)],
            'mean_diversity': [seq_diversity],
            'mean_novelty': [np.mean(seq_novelty)],
            'median_novelty': [np.median(seq_novelty)],
        })
        return results_df, metrics_df

def process_ggs_seqs(samples_path, sampling_method, topk, epoch_filter):
    """Process ggs samples."""
    generated_pairs = pd.read_csv(samples_path)
    generated_pairs = generated_pairs.drop_duplicates(
        subset='mutant_sequences', ignore_index=True)
    if epoch_filter is not None:
        generated_pairs = generated_pairs[generated_pairs.epoch <= epoch_filter]
    
    if sampling_method == 'greedy':
        generated_pairs = generated_pairs.sort_values(
            'mutant_scores', ascending=False)
        sampled_seqs = generated_pairs.mutant_sequences.tolist()[:topk]
        log.info(f'Sampled {len(set(sampled_seqs))} unique sequences.')
    else:
        raise ValueError(f'Bad sampling method: {sampling_method}')
    return sampled_seqs

def process_baseline_seqs(samples_path, topk):
    """Process baseline samples."""
    sampled_seqs = pd.read_csv(samples_path).sequence.tolist()
    if len(sampled_seqs) != topk:
        raise ValueError(f'Bad number of sequences {len(sampled_seqs)} != {topk}')
    return sampled_seqs

@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    exp_cfg = cfg.experiment

    # Set-up paths.
    method = exp_cfg.method
    if method == 'baselines':
        samples_dir = exp_cfg.baselines_samples_dir
        _method_fn = lambda x: process_baseline_seqs(x, exp_cfg.topk)
    elif method == 'ggs':
        samples_dir = exp_cfg.ggs_samples_dir
        _method_fn = lambda x: process_ggs_seqs(x, exp_cfg.topk_sampling, exp_cfg.topk, exp_cfg.epoch_filter)
    else:
        raise ValueError('Bad method')
    task = exp_cfg.task
    results_dir = os.path.join(samples_dir, exp_cfg.results_name)
    if method == 'ggs' and exp_cfg.epoch_filter is not None:
        log.info(f'Filtering up to epoch {exp_cfg.epoch_filter}')
        results_dir = os.path.join(results_dir, f'epoch_filter_{exp_cfg.epoch_filter}')

    os.makedirs(results_dir, exist_ok=True)
    log.info(f'Results saved to {results_dir}')

    # Set-up main class for running evaluation.
    # Hacky but it works...
    print(samples_dir)
    num_mutations = [
        x for x in samples_dir.split('/') if 'mutations' in x][0]
    starting_range = [
        x for x in samples_dir.split('/') if 'percentile' in x][0]
    if cfg.runner.base_pool_path is not None:
        raise ValueError(f'Expected base pool path to be None, got {cfg.runner.base_pool_path}')
    cfg.runner.base_pool_path = os.path.join(
        cfg.paths.data_dir, task, num_mutations, starting_range,
        'base_seqs.csv')
    eval_runner = EvalRunner(cfg.runner)

    # Glob results to evaluate.
    all_csv_paths = [
        path for path in glob.glob(os.path.join(samples_dir, '*.csv'))
        if 'aggregate' not in os.path.basename(path)
    ]
    log.info(f'Evaluating {len(all_csv_paths)} results in {samples_dir}')

    # Run evaluation for each result.
    all_results = []
    all_metrics = []
    use_oracle = False if cfg.runner.predictor_dir is not None else True
    for csv_path in tqdm(all_csv_paths):
        csv_path = os.path.join(results_dir, csv_path)
        topk_seqs = _method_fn(csv_path)
        csv_results, csv_metrics = eval_runner.evaluate_sequences(topk_seqs, use_oracle=use_oracle)
        log.info(f'Results for {csv_path}\n{csv_metrics}')
        csv_results['source_path'] = csv_path
        csv_metrics['source_path'] = csv_path
        all_results.append(csv_results)
        all_metrics.append(csv_metrics)
    all_results = pd.concat(all_results) 
    all_metrics = pd.concat(all_metrics)

    # Save results.
    output_fname =  f'results_oracle_{cfg.runner.oracle}' if use_oracle else 'results_predictor'
    if not cfg.runner.use_normalization:
        output_fname = output_fname + '_unnormalized'
    if method == 'ggs':
        output_fname =  output_fname + f'_sampling_{exp_cfg.topk_sampling}'
    output_fname = output_fname + '.csv'
    results_path = os.path.join(results_dir, output_fname)
    all_results.to_csv(results_path, index=False)
    log.info(f'Results written to {results_path}')

    # Save metrics.
    metrics_fname = output_fname.replace('results', 'metrics')
    metrics_path = os.path.join(results_dir, metrics_fname)
    all_metrics.to_csv(metrics_path, index=False)
    log.info(f'Metrics written to {metrics_path}')

    return all_results

if __name__ == "__main__":
    main()
