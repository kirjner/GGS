from typing import Any, Dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import logging
import os




class ProtFunctionTrainset(Dataset):

    def __init__(
        self,
        task: str,
        csv_dir: str,
        difficulty: Dict[str, Any],
        smoothing: str,
        alphabet: str,
        encoding: str,
        debug: bool = False
        ):

        self._log = logging.getLogger(__name__)
        self._debug = debug

        self.q_lb = difficulty.quantile_lb
        self.q_ub = difficulty.quantile_ub
        self.min_mutation_gap = difficulty.gap
        self.smoothing = smoothing
        self.alphabet = alphabet
        
        if encoding != 'onehot':
            raise ValueError(f"only one-hot encoding is currently implemented")
        
        self.encode = self._one_hot_encode 

        csv_path = os.path.join(csv_dir, f'range_{self.q_lb}_{self.q_ub}_gap_{self.min_mutation_gap}.csv')
        self.csv_path = csv_path if self.smoothing == 'unsmoothed' else csv_path.replace('.csv', '_smoothed.csv')
        if os.path.exists(self.csv_path):
            self._log.info(f"Loading {'unsmoothed' if self.smoothing== 'unsmoothed' else 'smoothed'} training set for task: {task}, with range: [{self.q_lb}, {self.q_ub}] and gap {self.min_mutation_gap}")
        else:
            raise ValueError(f"Training set for task {task}, with range: [{self.q_lb}, {self.q_ub}] and gap {self.min_mutation_gap} not found at {self.csv_path}. Run make_train_set.py (or GS.py) to generate the unsmoothed (or smoothed) training set.")
        self._data_df = pd.read_csv(self.csv_path)
        
        self._log.info(f"Loaded {len(self._data_df)} examples from {self.csv_path}")
        if self._debug:
            self._data_df = self._data_df.sample(frac=0.1)
            self._log.info(f"Debug mode: subsampling to {len(self._data_df)} examples")
        

    def _one_hot_encode(self, seq):
        return np.array([self.alphabet.index(x) for x in seq])

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, idx):
        row = self._data_df.iloc[idx]
        features = self.encode(row.sequence)
        target = float(row.score)
        return features, target

class ProtFunctionTaskDM(LightningDataModule):

    def __init__(
            self,
            *,
            dataset: Dict[str, Any],
            dataloader: Dict[str, Any],
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self._trainset = ProtFunctionTrainset(**dataset)
        self._data_loader_cfg = dataloader
        # Data paths

    def train_dataloader(self):
        return DataLoader(self._trainset, **self._data_loader_cfg)


