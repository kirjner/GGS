import torch
import numpy as np
import logging
import time
import torch.distributions as dists
import torch.nn.functional as F
import pandas as pd
from ggs.models.predictors import BaseCNN
from omegaconf import OmegaConf
import os
from ggs.data.utils.tokenize import Encoder

from typing import List

to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()

class GwgPairSampler(torch.nn.Module):
    
    def __init__(
            self,
            predictor_dir: str,
            temperature: float,
            ckpt_name: str,
            verbose: bool = False,
            gibbs_samples: int = 500,
            device: str = "cuda",
        ):
        super().__init__()
        self._ckpt_name = ckpt_name
        self._log = logging.getLogger(__name__)
        self.device = torch.device(device)
        self._log.info(f'Using device: {self.device}')
        self.predictor_tokenizer =Encoder()
        self.predictor = self._setup_predictor(predictor_dir)
        self.num_tokens = len(self.predictor_tokenizer.alphabet)
        self.temp = temperature
        self.total_pairs = 0
        self.num_current_src_seqs = 0
        self.gibbs_samples = gibbs_samples
        self._verbose = verbose
        self.sampled = 0
        self.accepted = 0

    def _setup_predictor(self, predictor_dir: str):
        # Load model weights.
        predictor_path = os.path.join(predictor_dir, self._ckpt_name)
        mdl_info = torch.load(predictor_path, map_location=self.device)
        cfg_path = os.path.join(predictor_dir, 'config.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)
        predictor = BaseCNN(make_one_hot=False, **ckpt_cfg.model.predictor)
        state_dict = {k.replace('predictor.', ''): v for k, v in mdl_info['state_dict'].items()}
        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(self.device)
        self._log.info(predictor)
        return predictor

    def tokenize_seqs(self, seqs):
        return self.gen_tokenizer.encode(seqs)
    

    def _calc_local_diff(self, seq_one_hot):
        # Construct local difference
        gx = torch.autograd.grad(self.predictor(seq_one_hot).sum(), seq_one_hot)[0]
        gx_cur = (gx * seq_one_hot).sum(-1)[:, :, None]
        delta_ij = gx - gx_cur
        return delta_ij

    
    def _gibbs_sampler(self, seq_one_hot):
        delta_ij = self._calc_local_diff(seq_one_hot)
        delta_ij = delta_ij[0]
        # One step of GWG sampling.
        def _gwg_sample():
            seq_len, num_tokens = delta_ij.shape
            # Construct proposal distributions
            gwg_proposal = dists.OneHotCategorical(logits = delta_ij.flatten() / self.temp)
            r_ij = gwg_proposal.sample((self.gibbs_samples,)).reshape(
                self.gibbs_samples, seq_len, num_tokens)

            # [num_samples, L, 20]
            seq_token = torch.argmax(seq_one_hot, dim=-1)
            mutated_seqs = seq_token.repeat(self.gibbs_samples, 1)
            seq_idx, res_idx, aa_idx = torch.where(r_ij)
            mutated_seqs[(seq_idx, res_idx)] = aa_idx
            return mutated_seqs
        
        return _gwg_sample


    def _make_one_hot(self, seq, differentiable=False):
        seq_one_hot = F.one_hot(seq, num_classes=self.num_tokens)
        if differentiable:
            seq_one_hot = seq_one_hot.float().requires_grad_()
        return seq_one_hot

    def _evaluate_one_hot(self, seq):
        input_one_hot = self._make_one_hot(seq)
        model_out = self.predictor(input_one_hot)
        return model_out

    def _decode(self, one_hot_seq):
        return self.predictor_tokenizer.decode(one_hot_seq)

    def _metropolis_hastings(
            self, mutants, source_one_hot, delta_score):
       
        source = torch.argmax(source_one_hot, dim=-1)
    
        # [num_seq, L]
        mutated_indices = mutants != source[None]
        # [num_seq, L, 20]
        mutant_one_hot = self._make_one_hot(mutants, differentiable=True)
        mutated_one_hot = mutant_one_hot * mutated_indices[..., None]
        
        source_delta_ij = self._calc_local_diff(source_one_hot[None])
        mutant_delta_ij = self._calc_local_diff(mutant_one_hot)

        orig_source_shape = source_delta_ij.shape
        orig_mutant_shape = mutant_delta_ij.shape

        # Flatten starting from the second to last dimension and apply softmax
        q_source = source_delta_ij.flatten(start_dim=-2)
        q_source = F.softmax(q_source / self.temp, dim=-1)

        q_mutant = mutant_delta_ij.flatten(start_dim=-2)
        q_mutant = F.softmax(q_mutant / self.temp, dim=-1)

        # Reshape back to the original shape
        q_source = q_source.view(orig_source_shape).squeeze(0)
        q_mutant = q_mutant.view(orig_mutant_shape)
        
        mutation_tuple = torch.nonzero(mutated_one_hot, as_tuple=True)
        q_ij_source = q_source[mutation_tuple[1], mutation_tuple[2]]
        q_ij_mutant = q_mutant[torch.arange(q_mutant.shape[0]).to(self.device), mutation_tuple[1], mutation_tuple[2]] 
        q_ij_ratio = q_ij_mutant / q_ij_source
        accept_prob = torch.exp(delta_score)*q_ij_ratio.to(self.device)
        
        mh_step = accept_prob < torch.rand(accept_prob.shape).to(self.device)
        return mh_step

    def _evaluate_mutants(
            self,
            *,
            mutants,
            score,
            source_one_hot,
        ):
        all_mutated_scores = self._evaluate_one_hot(mutants)
        delta_score = all_mutated_scores - score

        accept_mask = self._metropolis_hastings(
            mutants, source_one_hot, delta_score) 
        accepted_x = to_list(mutants[accept_mask])
        accepted_seq = [self._decode(x) for x in accepted_x]
        accepted_score = to_list(all_mutated_scores[accept_mask])
        return pd.DataFrame({
            'mutant_sequence': accepted_seq,
            'mutant_score': accepted_score,
        }), mutants[accept_mask]

    def compute_mutant_stats(self, source_seq, mutant_seqs):
        num_mutated_res = torch.sum(
            ~(mutant_seqs == source_seq[None]), dim=-1)
        return num_mutated_res

    def forward(self, batch):
        seqs = batch['sequence']
        #Tokenize
        tokenized_seqs = self.predictor_tokenizer.encode(seqs).to(self.device)
        total_num_seqs = len(tokenized_seqs)

        # Sweep over hyperparameters
        all_mutant_pairs = []
        grand_total_num_proposals = 0
        grand_total_num_accepts = 0
        for i, (real_seq, token_seq) in enumerate(zip(seqs, tokenized_seqs)):

            # Cast as float to take gradients through
            seq_one_hot = self._make_one_hot(token_seq, differentiable=True)

            # Compute base score
            pred_score = self._evaluate_one_hot(token_seq[None]).item()

            # Construct Gibbs sampler
            sampler = self._gibbs_sampler(seq_one_hot[None]) 
            seq_pairs = []
            total_num_proposals = 0
            all_proposed_mutants = []
            all_accepted_mutants = []

            # Sample mutants
            proposed_mutants = sampler()
            num_proposals = proposed_mutants.shape[0]
            total_num_proposals += num_proposals
            grand_total_num_proposals += num_proposals
            proposed_num_edits = self.compute_mutant_stats(
                token_seq, proposed_mutants)
            proposed_mutants = proposed_mutants[proposed_num_edits > 0]
            all_proposed_mutants.append(to_np(proposed_mutants))

            # Run Gibbs generation of pairs
            sample_outputs, accepted_mutants = self._evaluate_mutants(
                mutants=proposed_mutants,
                score=pred_score,
                source_one_hot=seq_one_hot
            )

            all_accepted_mutants.append(to_np(accepted_mutants))
            grand_total_num_accepts += len(accepted_mutants)
            sample_outputs['source_sequence'] = real_seq
            sample_outputs['source_score'] = pred_score

            seq_pairs.append(sample_outputs)
            if self._verbose:
                num_pairs = len(sample_outputs)
                print(
                    f'Temp: {self.temp:.3f}'
                    f'Accepted: {num_pairs}/{num_proposals} ({num_pairs/num_proposals:.2f})'
                )

            if len(seq_pairs) > 0:
                seq_pairs = pd.concat(seq_pairs).drop_duplicates(
                    subset=['source_sequence', 'mutant_sequence'],
                    ignore_index=True
                )
                all_mutant_pairs.append(seq_pairs)
        if self._verbose:
            print("Epoch acceptance rate: ", grand_total_num_accepts / grand_total_num_proposals)

        if len(all_mutant_pairs) == 0:
            return None
        return pd.concat(all_mutant_pairs).drop_duplicates(
            subset=['source_sequence', 'mutant_sequence'],
            ignore_index=True
        ), grand_total_num_accepts / grand_total_num_proposals
