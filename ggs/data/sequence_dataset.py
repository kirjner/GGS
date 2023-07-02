import logging
import pandas as pd
import numpy as np
from typing import List
import os
import torch
from torch.utils.data import Dataset
from ggs.data.utils.preprocessing import get_negative_aug_data
from polyleven import levenshtein
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm
tqdm.pandas()
from datetime import datetime

class SequenceDataset(Dataset):
    def __init__(
            self,
            *,
            csv_path: str,
            output_dir: str,
            filter_percentile: str,
            min_mutant_dist: int,
            top_quantile: float,
            use_neg_data: bool,
            alphabet: str,
            seed: int,
            smoothed_data_fname: str,
            use_levenshtein: bool,
            sequence_column: str,
            val_samples: int
        ):
        self._log = logging.getLogger(__name__)
        self._log.info(f"Reading CSV file {csv_path}")
        self._raw_data_df = pd.read_csv(csv_path)

        self._top_quantile = top_quantile
        self._alphabet = alphabet
        self._use_levenshtein = use_levenshtein
        self._sequence_column = sequence_column
        true_data = self._raw_data_df[self._raw_data_df.augmented == 0].copy()
        true_data['source'] = 'true'
        neg_data = self._raw_data_df[self._raw_data_df.augmented == 1].copy()
        neg_data['source'] = 'neg_aug'

        prev_num_rows = true_data.shape[0]
        filtered_data = self._filter(true_data, filter_percentile, min_mutant_dist)
        new_num_rows = filtered_data.shape[0]
        self._log.info(
            f"Filtered {prev_num_rows} to {new_num_rows} rows in {filter_percentile} "
            + f"score range and >={min_mutant_dist} mutations away.")

        percentile_str = '_'.join([str(x) for x in filter_percentile])
        
        write_dir = os.path.join(
            output_dir, f'mutations_{min_mutant_dist}', f'percentile_{percentile_str}'
        )
        
        write_path = os.path.join(
            write_dir, f'base_seqs.csv'
        )
        os.makedirs(write_dir, exist_ok=True)
        self._write_path = write_path
        filtered_data.to_csv(self._write_path, index=False)

        all_data = [
            filtered_data
        ]

        if smoothed_data_fname is not None:
            smoothed_path = os.path.join(write_dir, smoothed_data_fname)
            self._log.info(f'Using smoothed data from {smoothed_path}')
            if not os.path.exists(smoothed_path):
                raise ValueError(f"Could not find smoothed data at {smoothed_path}")
            smoothed_data = pd.read_csv(smoothed_path)
            smoothed_data['source'] = 'smoothed'
            all_data.append(smoothed_data)
            self._log.info(f'Read in {len(smoothed_data)} smoothed sequences.')

        if use_neg_data:
            max_neg_samples = filtered_data.shape[0]
            if len(neg_data) >= max_neg_samples:
                neg_data = neg_data.sample(max_neg_samples, random_state=seed)
            else:
                seq_len = len(filtered_data[sequence_column].iloc[0])
                if 'AAV' in csv_path:
                    neg_aug_val = 0
                else:
                    neg_aug_val = 1
                neg_data = get_negative_aug_data(num_neg_samples=max_neg_samples,
                                                 neg_aug_dict={'value': neg_aug_val,
                                                               'method': 'random'},
                                                 sample_length=seq_len, alphabet=alphabet)
            all_data.append(neg_data)

        self._data_df = pd.concat(all_data)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self._seq_len = len(self._data_df[sequence_column].iloc[0])
        self._log.info(f"Preprocessed data has {len(self._data_df)} rows.")

    def get_source_indices(self, source):
        return np.where(self._data_df.source == source)[0]

    def _filter(self, data_df, percentile, min_mutant_dist):
        lower_value = data_df.target.quantile(percentile[0])
        upper_value = data_df.target.quantile(percentile[1])
        top_quantile = data_df.target.quantile(self._top_quantile)
        top_sequences_df = data_df[data_df.target >= top_quantile]  
        self._log.info('Filtering')
        filtered_df = data_df[data_df.target.between(lower_value, upper_value)]
        if self._use_levenshtein:
            get_min_dist = lambda x: np.min([levenshtein(x.strip(), top_seq.strip()) for top_seq in top_sequences_df.sequence]) 
            self._log.info('Getting minimum Levenshtein distance to top sequences')
            mutant_dist = filtered_df.sequence.progress_map(get_min_dist)
        else:
            top_sequences = np.stack(
                [self._encode_sequence(x) for x in top_sequences_df[self._sequence_column]])
            get_min_dist = lambda x: np.min(
                np.sum(self._encode_sequence(x)[None] != top_sequences, axis=-1))
            mutant_dist = filtered_df[self._sequence_column].apply(get_min_dist)
        return filtered_df[mutant_dist >= min_mutant_dist]
    
    def _encode_sequence(self, seq):
        return np.array([self._alphabet.index(x) for x in seq])

    def extend(self, new_seqs, new_features, new_targets):
        #First, binary array indicating old vs new seq idxs

        self.extended = True
        self.is_new = np.array([0]*len(self.seqs) + [1]*len(new_seqs), dtype=bool)
        self.seqs = np.concatenate((self.seqs, new_seqs))
        self.features = torch.cat((self.features, torch.tensor(new_features)), axis = 0)
        self.targets = torch.cat((self.targets, torch.tensor(new_targets)))

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, idx):
        row = self._data_df.iloc[idx]
        seq = row[self._sequence_column]
        features = self._encode_sequence(seq)
        target = float(row.target)
        return features, target


class FoldingSequenceDataset(Dataset): #NOTE: Not currently supported
    def __init__(
            self,
            *,
            csv_path: str,
            output_dir: str,
            filter_range: List[int],
            min_mutant_dist: int,
            top_fitness: float,
            num_base_samples: int,
            alphabet: str,
            seed: int,
        ):
        self._log = logging.getLogger(__name__)
        self._log.info(f"Reading CSV file {csv_path}")
        self._raw_data_df = pd.read_csv(csv_path)
        self._alphabet = alphabet

        # Filtered data paths.
        percentile_str = '_'.join([str(x) for x in filter_range])
        write_path = os.path.join(
            output_dir,  # Back out two levels.
            f'mutations_{min_mutant_dist}',
            f'percentile_{percentile_str}',
            f'base_seqs_sample_{num_base_samples}_seed_{seed}.csv'
        )
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        self._write_path = write_path
        if os.path.exists(self._write_path):
            filtered_data = pd.read_csv(self._write_path)
            self._log.info(f'Read filtered data from {self._write_path}')
        else:
            # Filter sequences on their fitness and distance to the best sequences.
            prev_num_rows = self._raw_data_df.shape[0]
            filtered_data = self._filter(
                data_df=self._raw_data_df,
                filter_range=filter_range,
                min_mutant_dist=min_mutant_dist,
                top_fitness=top_fitness
            )
            new_num_rows = filtered_data.shape[0]
            self._log.info(
                f"Filtered {prev_num_rows} to {new_num_rows} rows in {filter_range} "
                + f"score range and >{min_mutant_dist} mutations away.")
            if num_base_samples is not None:
                filtered_data = filtered_data.sample(
                    num_base_samples, random_state=seed)
            filtered_data.to_csv(self._write_path, index=False)

        self._data_df = filtered_data
        self._log.info(f"Preprocess data has {len(self._data_df)} rows.")

    def _filter(self, *, data_df, filter_range, min_mutant_dist, top_fitness):
        
        # Filter based on fitness.
        filtered_df = data_df[data_df.fitness.between(filter_range[0], filter_range[1])]

        # Get distance to top sequences
        top_seqs = data_df[data_df.fitness < top_fitness]
        top_sequences = np.stack(
            [self._encode_sequence(x) for x in top_seqs.sequence])
        candidate_sequences = np.stack(
            [self._encode_sequence(x) for x in filtered_df.sequence])
        dist = np.sum(
            candidate_sequences[:, None, :] != top_sequences[None, :, :],
            axis=-1
        )
        min_dist = np.min(dist, axis=-1)
        filtered_df['min_dist'] = min_dist
        return filtered_df[min_dist >= min_mutant_dist]
    
    def _encode_sequence(self, seq):
        return np.array([self._alphabet.index(x) for x in seq])

    def extend(self, new_seqs, new_features, new_targets):
        #First, binary array indicating old vs new seq idxs

        self.extended = True
        self.is_new = np.array([0]*len(self.seqs) + [1]*len(new_seqs), dtype=bool)
        self.seqs = np.concatenate((self.seqs, new_seqs))
        self.features = torch.cat((self.features, torch.tensor(new_features)), axis = 0)
        self.targets = torch.cat((self.targets, torch.tensor(new_targets)))

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, idx):
        row = self._data_df.iloc[idx]
        seq = row.sequence
        features = self._encode_sequence(seq)
        target = float(row.fitness)
        return features, target

class PreScoredSequenceDataset(Dataset):
    """_summary_
    Args: data_dir (str): path to data directory
        csv_file (str): path to csv file

    Returns:
        _type_: _description_
    """

    def __init__(
            self,
            *,
            csv_path,
            cluster_cutoff,
            max_visits,
        ):
        self._log = logging.getLogger(__name__)
        self._log.info(f"Reading csv file from {csv_path}")
        self._raw_data = pd.read_csv(csv_path).rename(
            columns={'target': 'scores', 'sequence': 'sequences'})
        self._data = self._raw_data.copy()
        self._log.info(
            f"Found {len(self.sequences)} sequences "
            f"with TRUE scores between {np.min(self.scores):.2f} and {np.max(self.scores):.2f}"
        )
        self._cluster_cutoff = cluster_cutoff
        self._log.info('Clustering with TRUE scores. After this everything is predicted scores.')
        self.cluster()
        self._observed_sequences = {seq: 1 for seq in self.sequences}
        self._max_visits = max_visits
        self._pairs = pd.DataFrame({
            'source_sequences': [],
            'mutant_sequences': [],
            'source_scores': [],
            'mutant_scores': [],
            'epoch': [],
        })

    @property
    def sequences(self):
        return self._data.sequences.tolist()

    @property
    def scores(self):
        return self._data.scores.tolist()
    
    @property
    def pairs(self):
        return self._pairs

    def add_pairs(self, new_pairs, epoch):
        prev_num_pairs = len(self._pairs)
        new_pairs['epoch'] = epoch
        updated_pairs = pd.concat([self._pairs, new_pairs])
        updated_pairs = updated_pairs.drop_duplicates(
            subset=['source_sequences', 'mutant_sequences'], ignore_index=True)
        num_new_pairs = len(updated_pairs) - prev_num_pairs
        self._log.info(f'Added {len(updated_pairs) - prev_num_pairs} pairs.')
        self._pairs = updated_pairs
        return num_new_pairs

    def get_visits(self, sequences):
        return [
            self._observed_sequences[seq] if seq in self._observed_sequences else 0
            for seq in sequences
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx): 
        row = self._data.iloc[idx]
        return {
            'sequences': row['sequences'],
            'scores': row['scores'],
        }

    def cluster(self):
        # Convert to integer array. Doesn't matter what ordering we use.
        alphabet = "ARNDCQEGHILKMFPSTWYV"
        seq_ints = [[
            alphabet.index(x) for x in seq
        ] for seq in self.sequences]
        seq_array = np.array(seq_ints)
        Z = linkage(seq_array, 'average', metric='hamming')

        # Cluster to desired number of clusters.
        cluster_assignments = fcluster(Z, t=self._cluster_cutoff, criterion='maxclust')

        # Update Dataframe
        prev_num_seqs = len(self.sequences)
        self._data['cluster'] = cluster_assignments.tolist()
        max_cluster_fitness = {}
        for cluster, cluster_df in self._data.groupby('cluster'):
            max_cluster_fitness[cluster] = cluster_df['scores'].max()
        self._data = self._data[
            self._data.apply(
                lambda x: x.scores == max_cluster_fitness[x.cluster], axis=1
            )
        ]
        self._log.info(
            f"Clustered {prev_num_seqs} sequences to {len(self.sequences)} sequences "
            f"with scores min={np.min(self.scores):.2f}, max={np.max(self.scores):.2f}, "
            f"mean={np.mean(self.scores):.2f}, std={np.std(self.scores):.2f}"
        )

    def remove(self, seqs):
        """Remove sequence(s) and score(s)."""
        if not isinstance(seqs, list):
            seqs = [seqs]
        if len(seqs) == 0:
            return
        prev_num_seqs = len(self.sequences)
        self._data = self._data[~self._data.sequence.isin(seqs)]
        removed_num_seqs = len(self.sequences) - prev_num_seqs
        self._log.info(f"Removed {removed_num_seqs} sequences.")

    def reset(self):
        self._data = pd.DataFrame(columns=self._data.columns)

    def add(self, new_seqs):
        """Add sequence(s) and score(s) to the end of the dataset"""
        filtered_seqs = new_seqs[np.array(self.get_visits(new_seqs.sequences)) < self._max_visits]
        prev_num_seqs = len(self.sequences)
        self._data = pd.concat([self._data, filtered_seqs])
        self._data = self._data.drop_duplicates(subset=['sequences'], ignore_index=True)
        added_num_seqs = len(self._data) - prev_num_seqs
        self._log.info(f"Added {added_num_seqs} sequences.")
