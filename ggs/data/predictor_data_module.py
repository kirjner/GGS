from typing import Any, Dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset, Subset
from ggs.data.sequence_dataset import SequenceDataset
import logging
import random

class PandasDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sequence = self.dataset.iloc[index]['sequence']
        target = self.dataset.iloc[index]['target']
        return sequence, target

    def __len__(self):
        return len(self.dataset)

class PredictorDataModule(LightningDataModule):

    def __init__(
            self,
            *,
            task: str,
            task_cfg: Dict[str, Any],
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            encoding: str, # how to prepare the fasta file for the model
            alphabet: str, # amino acid alphabet
            val_samples: float,
            seed: int,
            sequence_column: str,
            weighted_sampling: bool
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        # Data paths
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._encoding = encoding
        self._seed = seed
        self._weighted_sampling = weighted_sampling
        if task in {'GFP', 'AAV'}:
            self._dataset = SequenceDataset(
                **task_cfg,
                alphabet=alphabet,
                seed=self._seed,
                sequence_column=sequence_column,
                val_samples=val_samples,
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
        self._log.info(f'Dataset: {len(self._dataset)} examples from the screen')

    
    def train_dataloader(self):
        sampler = None
        if self._weighted_sampling:
            '''
            If we are performing weighted sampling, we assume the weight of an example are inversely proportional value of that example's score
            Target values can be negative, so we add the minimum score value to all scores to make them positive
            '''
            self._log.info('Using weighted sampling')
            targets = self._dataset._data_df.score
            adjusted_targets = targets - targets.min() + 1
            weights = 1 / adjusted_targets.values
            sampler = WeightedRandomSampler(weights, len(weights))
        # torch_dataset = data.DataFrame(X=self._dataset._data_df.drop('target',axis=1),
        #                                y=self._dataset._data_df.score)

        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            sampler=sampler
        )

