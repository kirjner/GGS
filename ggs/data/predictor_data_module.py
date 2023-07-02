from typing import Any, Dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from ggs.data.sequence_dataset import SequenceDataset
import logging
import random
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
            sequence_column: str
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        # Data paths
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._encoding = encoding
        self._seed = seed
        if task in {'GFP', 'AAV'}:
            self._dataset = SequenceDataset(
                **task_cfg,
                alphabet=alphabet,
                seed=self._seed,
                sequence_column=sequence_column,
                val_samples=val_samples,
            )
        elif task == 'folding':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown task: {task}")
        
        true_indices = self._dataset.get_source_indices('true')
        valid_indices = random.sample(true_indices.tolist(), int(val_samples))
        train_indices = set(range(len(self._dataset))) - set(valid_indices)
        train_indices = list(train_indices)
        self.train_dataset = Subset(self._dataset, train_indices)
        self.val_dataset = Subset(self._dataset, valid_indices)
        self._log.info(f'Train dataset: {len(self.train_dataset)} examples')
        self._log.info(f'Train dataset has {len(train_indices) - self._dataset._data_df.iloc[train_indices].augmented.sum()} examples from ground truth')
        self._log.info(f'Train dataset: { self._dataset._data_df.iloc[train_indices].augmented.sum()} augmented examples')
        train_df = self._dataset._data_df.iloc[train_indices]
        self._log.info(f'Average Augmented Value: {train_df[train_df.augmented == 1].target.mean()}')
        self._log.info(f'Validation dataset: {len(self.val_dataset)} examples from ground truth')

    def _create_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=shuffle,
        )
    
    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)
