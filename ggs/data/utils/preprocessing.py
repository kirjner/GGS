from typing import Optional
from tqdm.auto import tqdm
from ggs.data.utils.tokenize import Encoder
import torch
import numpy as np
import pandas as pd


def encode_sequence_data(sequences: list[str], scores: list[float], alphabet: str, neg_aug_dict: Optional[dict] = None):
    data = []
    encoder =Encoder(alphabet)
    for sequence, score in tqdm(zip(sequences, scores)):
        x = list(encoder.encode(sequence))
        if isinstance(x[0], torch.Tensor):
            x = [x_i.item() for x_i in x]
        data.append([sequence] + x + [score])
    column_names = ['sequence'] + ['x' + str(i) for i in range(len(x))] + ['target']
    if neg_aug_dict is None or not neg_aug_dict['use']:
        positive_df = pd.DataFrame(data, columns=column_names)
        return positive_df

    column_names += ['augmented']
    augmented = [0] * len(data)
    data = [data[i] + [augmented[i]] for i in range(len(data))]
    positive_df = pd.DataFrame(data, columns=column_names)

    sample_length = len(data[0][0])  # sequence length
    num_neg_samples = len(data)
    negative_df = get_negative_aug_data(num_neg_samples, neg_aug_dict, sample_length, alphabet)
    return pd.concat((positive_df, negative_df))


def get_negative_aug_data(num_neg_samples: int, neg_aug_dict: dict, sample_length: int, alphabet: str):
    neg_target = neg_aug_dict['value']
    neg_targets = np.random.normal(neg_target, 0.1, num_neg_samples)  # same number as in training set
    encoder =Encoder(alphabet)
    data = []
    if neg_aug_dict['method'] == 'random':
        samples = np.random.choice(list(alphabet), size=(num_neg_samples, sample_length))
        samples = [''.join(sample) for sample in samples]
    else:
        raise NotImplementedError
    for i, sample in enumerate(samples):
        x = list(encoder.encode(sample))
        if isinstance(x[0], torch.Tensor):
            x = [x_i.item() for x_i in x]
        data.append([sample] + x + [neg_targets[i]] + [1])  # 1 means negative sample
    column_names = ['sequence'] + ['x' + str(i) for i in range(len(x))] + ['target', 'augmented']
    return pd.DataFrame(data, columns=column_names)