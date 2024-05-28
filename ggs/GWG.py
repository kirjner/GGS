from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as L
import pyrootutils
import copy
import time
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import pandas as pd
import torch
from ggs.models.GWG_module import GwgPairSampler
from ggs.data.sequence_dataset import PreScoredSequenceDataset
import pickle as pkl

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from ggs import utils

log = utils.get_pylogger(__name__)
to_list = lambda x: x.cpu().detach().numpy().tolist()

def save_pairs(pairs_dict, save_path):
    """Save generated pairs to file"""
    log.info(f"Saving generated pairs to {save_path}")
    with open(save_path, "w") as f:
        for src_seq in pairs_dict:
            for tgt_seq in pairs_dict[src_seq]:
                f.write(f"{src_seq},{tgt_seq}\n")

def _worker_fn(args):
    """Worker function for multiprocessing.

    Args:
        args (tuple): (worker_i, exp_cfg, inputs)
            worker_i: worker id. Used for setting CUDA device.
            exp_cfg: model config.
            inputs: list of inputs to process.

    Returns:
        all_outputs: results of GWG.
    """
    worker_i, exp_cfg, inputs = args
    model = GwgPairSampler(
        **exp_cfg,
        device=f"cuda:0"
    )
    all_candidates, all_acceptance_rates = [], []
    for batch in inputs:
        candidates, acceptance_rate = model(batch)
        all_candidates.append(candidates)
        all_acceptance_rates.append(acceptance_rate)
    log.info(f'Done with worker: {worker_i}')
    return all_candidates, all_acceptance_rates

def _setup_dataset(cfg):
    if cfg.data.csv_path is not None:
        raise ValueError(f'cfg.data.csv_file must be None.')
    data_dir = os.path.dirname(cfg.experiment.predictor_dir).replace('ckpt', 'data')
    for i, subdir in enumerate(data_dir.split('/')):
        if 'percentile' in subdir:
            break
    data_dir = '/'.join(data_dir.split('/')[:i+1])
    cfg.data.csv_path = os.path.join(data_dir, 'base_seqs.csv')
    if not os.path.exists(cfg.data.csv_path):
        raise ValueError(f'Could not find dataset at {cfg.data.csv_path}.')
    return PreScoredSequenceDataset(**cfg.data)

def generate_pairs(cfg: DictConfig, sample_write_path: str) -> Tuple[dict, dict]:
    """Generate pairs using GWG."""
    cfg = copy.deepcopy(cfg)
    run_cfg = cfg.run

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    dataset = _setup_dataset(cfg)

    # Special settings for debugging.
    exp_cfg = dict(cfg.experiment)
    epoch = 0
    start_time = time.time()
    acceptance_rates = []
    while epoch < run_cfg.max_epochs and len(dataset):
        epoch += 1
        epoch_start_time = time.time()
        if run_cfg.debug:
            batch_size = 2
        else:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Run sampling with workers
        batches_per_worker = [[]]
        for i, batch in enumerate(dataloader):
            batches_per_worker[i].append(batch)
            if run_cfg.debug:
                break
        log.info(f"using GPU: {torch.device('cuda')}" )
        all_worker_outputs = [
            _worker_fn((0, exp_cfg, batches_per_worker[0]))
        ]

        # Process results.
        epoch_pair_count = 0
        candidate_seqs = []

        for worker_results, acceptance_rate in all_worker_outputs:
            for new_pairs in worker_results:
                if new_pairs is None:
                    continue
            
                candidate_seqs.append(
                    new_pairs[['mutant_sequence', 'mutant_score']].rename(
                        columns={'mutant_sequence': 'sequence', 'mutant_score': 'score'}
                    )
                )
            acceptance_rates.append(acceptance_rate[0])
            epoch_pair_count += dataset.add_pairs(new_pairs, epoch)
        if len(candidate_seqs) > 0:
            candidate_seqs = pd.concat(candidate_seqs)
            candidate_seqs.drop_duplicates(subset='sequence', inplace=True)
        epoch_elapsed_time = time.time() - epoch_start_time

        log.info(f"Epoch {epoch} finished in {epoch_elapsed_time:.2f} seconds")
        log.info("------------------------------------")
        log.info(f"Generated {epoch_pair_count} pairs in this epoch")
        log.info(f"Acceptance rate = {acceptance_rate[0]:.2f}")
        dataset.reset()
        if epoch < run_cfg.max_epochs and len(candidate_seqs) > 1:
            dataset.add(candidate_seqs)
            if cfg.data.clustering:
                dataset.cluster()
        log.info(f"Next dataset = {len(dataset)} sequences")
    dataset.pairs.to_csv(sample_write_path, index=False)
    # save acceptance_rates to pkl
    cluster_centers_path = sample_write_path.replace('.csv', '_cluster_centers.csv')
    dataset._cluster_centers.to_csv(cluster_centers_path, index=False)
    acceptance_rates_path = sample_write_path.replace('.csv', '_acceptance_rates.pkl')
    with open(acceptance_rates_path, 'wb') as f:
        pkl.dump(acceptance_rates, f)
        
    elapsed_time = time.time() - start_time
    log.info(f'Finished generation in {elapsed_time:.2f} seconds.')
    log.info(f'Samples written to {sample_write_path}.')


@hydra.main(version_base="1.3", config_path="../configs", config_name="GWG.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # Set-up output path
    run_cfg = cfg.run
    output_dir = os.path.join(cfg.experiment.predictor_dir, 'samples', run_cfg.run_name)
    cfg_write_path = os.path.join(output_dir, 'config.yaml')
    os.makedirs(os.path.dirname(cfg_write_path), exist_ok=True)
    with open(cfg_write_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    log.info(f'Config saved to {cfg_write_path}')

    # Generate samples for multiple seeds.
    seed = run_cfg.seed
    sample_write_path = os.path.join(output_dir, f'seed_{seed}.csv')
    log.info(f'On seed {seed}. Saving results to {sample_write_path}')
    generate_pairs(cfg, sample_write_path)

if __name__ == "__main__":
    main()
