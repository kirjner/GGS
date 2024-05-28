
from typing import List, Optional, Tuple
import hydra
import numpy as np
import pandas as pd
from ggs.models.predictors import BaseCNN
from random import sample
import random
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix, load_npz
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pyrootutils
import torch
from copy import deepcopy
import logging
import time
import os
from datetime import datetime
from pykeops.torch import Vi, Vj
from scipy.sparse.linalg import cg
from scipy.sparse import identity, csr_matrix, save_npz
from ggs.data.utils.tokenize import Encoder
from tqdm import tqdm
import sys
import pickle as pkl

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger('Graph-based Smoothing')
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

ALPHABET = list("ARNDCQEGHILKMFPSTWYV")
use_cuda = torch.cuda.is_available()
def tensor(*x):
    if use_cuda:
        return torch.cuda.FloatTensor(*x)
    else:
        return torch.FloatTensor(*x)

def KNN_KeOps(K, metric="euclidean", **kwargs):
    def fit(x_train):
        # Setup the K-NN estimator:
        x_train = tensor(x_train)
        #start = timer()

        # Encoding as KeOps LazyTensors:
        D = x_train.shape[1]
        X_i = Vi(0, D)  # Purely symbolic "i" variable, without any data array
        X_j = Vj(1, D)  # Purely symbolic "j" variable, without any data array

        # Symbolic distance matrix:
        if metric == "euclidean":
            D_ij = ((X_i - X_j) ** 2).sum(-1)
        elif metric == "manhattan":
            D_ij = ((X_i - X_j).abs()).sum(-1)
        elif metric == 'levenshtein':
            D_ij = (-((X_i-X_j).abs())).ifelse(0, 1).sum(-1)
        elif metric == "angular":
            D_ij = -(X_i | X_j)
        elif metric == "hyperbolic":
            D_ij = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])
        else:
            raise NotImplementedError(f"The '{metric}' distance is not supported.")

        # K-NN query operator:
        KNN_fun = D_ij.Kmin_argKmin(K, dim=1)

        def f(x_test):
            x_test = tensor(x_test)

            # Actual K-NN query:
            vals, indices  = KNN_fun(x_test, x_train)

            vals = vals.cpu().numpy()
            indices = indices.cpu().numpy()
            return vals, indices

        return f

    return fit

def run_predictor(seqs, batch_size, predictor):
    batches = torch.split(seqs, batch_size, 0)
    scores = []
    for b in batches:
        if b is None:
            continue
        results = predictor(b).detach()
        scores.append(results)
    return torch.concat(scores, dim=0)

def to_seq_tensor(seq):
    seq_ints = [
        ALPHABET.index(x) for x in seq
    ]
    return torch.tensor(seq_ints)


def get_next_state(seq, task, random_obj, num=1):
    seq_list = list(seq)
    seq_len = len(seq)
    position = random_obj.randint(0, seq_len-1)
    substitution = random_obj.choice(ALPHABET)
    seq_new = seq_list.copy()
    seq_new[position] = substitution
    return ''.join(seq_new)


def to_batch_tensor(seq_list, task, subset=None, device='cpu'):
    if subset is not None:
        seq_list = seq_list[:subset]
    return torch.stack([to_seq_tensor(x) for x in seq_list]).to(device)


def maximum(A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def extract_smoothing_params(smoothing_method):
    if 'tik-gamma' in smoothing_method:
        method = 'tik'
        gamma = float(smoothing_method.split('-')[-1])
    elif 'L2-gamma' in smoothing_method:
        method = 'L2'
        gamma = float(smoothing_method.split('-')[-1])
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}")
    return method, gamma

def generate_and_evaluate_sequences(cfg, df_base, task, device, predictor):
    start_time = time.time()
    max_n_seqs = cfg.max_n_seqs
    exploration_method = cfg.exploration_method
    if exploration_method == 'ham1':
        logger.info('Using ham1 strategy: generating sequences in 1-nbhd of base sequence pool..')
    original_seqs = list(set(df_base[cfg.sequence_columns].values))
    all_seqs_generated = original_seqs.copy()
    i_pointer = 0
    unique_seqs = set(original_seqs)
    random_seed = cfg.experiment.random_seed
    random_obj = random.Random(random_seed)
    pbar = tqdm(total=max_n_seqs, file=sys.stdout)
    
    '''
    NOTE: This generation procedure assumes that ham1 is the exploration method
    i.e. explore the 1-neighbors of the base sequences (the training set of the 
    unsmoothed model)
    '''
    while len(all_seqs_generated) < max_n_seqs:
        next_seq = all_seqs_generated[i_pointer]
        if next_seq not in original_seqs and exploration_method == 'ham1':
            print("WARNING: next_seq not in original_seqs")
            break
        new_seq = get_next_state(next_seq, task, random_obj)
        pbar.update(1)
        if new_seq not in unique_seqs:
            all_seqs_generated.append(new_seq)
            unique_seqs.add(new_seq)
            i_pointer = (i_pointer + 1) % len(original_seqs) if exploration_method == 'ham1' else i_pointer + 1
        if len(all_seqs_generated) >= 2 * max_n_seqs:
            break
            
    
    pbar.close()
    logger.info("Finished generating sequences..")
    all_seqs = list(sorted(set(all_seqs_generated)))
    all_seqs_encoded = tensor(Encoder(alphabet=ALPHABET).encode(all_seqs).to(torch.float).to(device))

    all_seqs_pt = to_batch_tensor(all_seqs, task, subset=None, device=device)
    node_scores_init = run_predictor(all_seqs_pt, batch_size=256, predictor=predictor).cpu().numpy()
    elapsed_time = time.time() - start_time
    logger.info(f'Finished generation + evaluation in {elapsed_time:.2f} seconds')
    
    logger.info('Creating KNN graph..')
    start_time = time.time()
    #sqrt of the number of sequences is a good heuristic for K
    fit_KeOps = KNN_KeOps(K=int(np.floor(np.sqrt(len(all_seqs)) + 1)), metric='levenshtein')(all_seqs_encoded)
    vals, indices = fit_KeOps(all_seqs_encoded)
    elapsed_time = time.time() - start_time
    logger.info(f'Finished kNN construction in {elapsed_time:.2f} seconds')
    
    vals = 1 / vals[:, 1:]
    indices = indices[:, 1:]
    non_mutual_knn_graph = csr_matrix((vals.flatten(), indices.flatten(), np.arange(0, len(vals.flatten()) + 1, len(vals[0]))))
    mutual_knn_graph = maximum(non_mutual_knn_graph, non_mutual_knn_graph.T)
    knn_graph = csr_matrix((1 / mutual_knn_graph.data, mutual_knn_graph.indices, mutual_knn_graph.indptr))
    
    logger.info('Computing Laplacian..')
    L = laplacian(knn_graph, normed=True).tocsr()    
    return all_seqs, node_scores_init, L

def compute_regularization_matrix(smoothing_method, gamma, L):
    I = identity(L.shape[0], format='csr')
    if smoothing_method == 'tik':
        return I + gamma * L
    elif smoothing_method == 'L2':
        return I + gamma * I
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}")

def store_results(cfg, data_dir, all_seqs, Y_opt, L):
    df_smoothed = pd.DataFrame({'sequence': all_seqs, 'score': Y_opt})
    now = datetime.now().strftime("%m_%d_%Y_%H_%M")
    results_dir = os.path.join(data_dir, cfg.smoothing_method, f"{cfg.exploration_method}_n-{cfg.max_n_seqs // 1000}K")
    os.makedirs(results_dir, exist_ok=True)
    if cfg.experiment.save_laplacian:
        laplacian_path = os.path.join(results_dir, f'laplacian-{now}.pkl')
        laplacian_dict = {'seqs': all_seqs, 'L': L}
        logger.info(f'Saving Laplacian and seqs to {laplacian_path} - WARNING: may be a large file')
        with open(laplacian_path, 'wb') as f:
            pkl.dump(laplacian_dict, f)
    results_filename = f'{cfg.results_file}-{now}' if cfg.results_file else f'smoothed-{now}'
    results_path = os.path.join(results_dir, f'{results_filename}.csv')
    logger.info(f'Writing results to {results_path}')
    df_smoothed.to_csv(results_path, index=None)
    
    cfg_write_path = os.path.join(results_dir, f'{results_filename}.yaml')
    with open(cfg_write_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)

@hydra.main(version_base="1.3", config_path="../configs", config_name="GS.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    predictor_dir = cfg.experiment.predictor_dir
    num_mutations = next(x for x in predictor_dir.split('/') if 'mutations' in x)
    starting_range = next(x for x in predictor_dir.split('/') if 'percentile' in x)
    
    if 'GFP' in predictor_dir:
        task = 'GFP'
    elif 'AAV' in predictor_dir:
        task = 'AAV'
    else:
        raise ValueError(f'Task not found in predictor path: {predictor_dir}')
    
    data_dir = os.path.join(cfg.paths.data_dir, task, num_mutations, starting_range)
    base_pool_path = os.path.join(data_dir, 'base_seqs.csv')
    df_base = pd.read_csv(base_pool_path)
    logger.info(f'Loaded base sequences {base_pool_path}')

    predictor_path = os.path.join(predictor_dir, cfg.ckpt_file)
    cfg_path = os.path.join(predictor_dir, 'config.yaml')
    with open(cfg_path, 'r') as fp:
        ckpt_cfg = OmegaConf.load(fp.name)
    predictor = BaseCNN(**ckpt_cfg.model.predictor)
    predictor_info = torch.load(predictor_path, map_location='cuda:0')
    predictor.load_state_dict({k.replace('predictor.', ''): v for k, v in predictor_info['state_dict'].items()}, strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor.to(device).eval()
    logger.info(f'Loading base predictor {predictor_path}')

    if cfg.experiment.laplacian_path:
        print("Using existing Laplacian matrix")
        with open(cfg.experiment.laplacian_path, 'rb') as f:
            laplacian_dict = pkl.load(f)
        L = laplacian_dict['L']
        all_seqs = laplacian_dict['seqs']
        all_seqs_pt = to_batch_tensor(all_seqs, task, subset=None, device=device)
        node_scores_init = run_predictor(all_seqs_pt, batch_size=256, predictor=predictor).cpu().numpy()
    else:
        all_seqs, node_scores_init, L = generate_and_evaluate_sequences(cfg, df_base, task, device, predictor)
    

    
    S_init = node_scores_init.copy()
    smoothing_method, gamma = extract_smoothing_params(cfg.smoothing_method)
    A = compute_regularization_matrix(smoothing_method, gamma, L)
    Y_opt, _ = cg(A, S_init)
    
    logger.info('Storing results..')
    store_results(cfg, data_dir, all_seqs, Y_opt, L)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main()


# def main(cfg: DictConfig) -> Optional[float]:

#     # Extract data path from predictor_dir
#     predictor_dir = cfg.experiment.predictor_dir
#     print(predictor_dir)
#     num_mutations = [
#         x for x in predictor_dir.split('/') if 'mutations' in x][0]
#     starting_range = [
#         x for x in predictor_dir.split('/') if 'percentile' in x][0]
#     if 'GFP' in predictor_dir:
#         task = 'GFP'
#     elif 'AAV' in predictor_dir:
#         task = 'AAV'
#         raise ValueError(f'Task not found in predictor path: {predictor_dir}')
    
    
#     data_dir = os.path.join(
#         cfg.paths.data_dir, task, num_mutations, starting_range)
#     base_pool_path = os.path.join(data_dir, 'base_seqs.csv')
#     df_base = pd.read_csv(base_pool_path)
#     logger.info(f'Loaded base sequences {base_pool_path}')

#     # Load predictor
#     predictor_path = os.path.join(predictor_dir, cfg.ckpt_file)
#     cfg_path = os.path.join(predictor_dir, 'config.yaml')
#     with open(cfg_path, 'r') as fp:
#         ckpt_cfg = OmegaConf.load(fp.name)
#     predictor = BaseCNN(**ckpt_cfg.model.predictor) 
#     predictor_info = torch.load(predictor_path, map_location='cuda:0')
#     predictor.load_state_dict({k.replace('predictor.', ''): v for k, v in predictor_info['state_dict'].items()}, strict=True)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     predictor.to(device).eval()
#     logger.info(f'Loading base predictor {predictor_path}')

#     if cfg.experiment.all_seqs_path is not None and cfg.experiment.laplacian_path is not None:
#         print("Using existing all_seqs and Laplacian matrix")
#         # Load existing all_seqs and Laplacian matrix
#         with open(cfg.experiment.all_seqs_path, 'rb') as f:
#             all_seqs_list_orig = pkl.load(f)
#         L = load_npz(cfg.experiment.laplacian_path)
#         all_seqs_pt = to_batch_tensor(all_seqs_list_orig, task, subset=None, device=device)
#         node_scores_init = run_predictor(all_seqs_pt, batch_size=256, predictor=predictor).cpu().numpy()
#     # Random walk
#     else:
#         max_n_seqs = cfg.max_n_seqs
#         start_time = time.time()
#         exploration_method = cfg.exploration_method
#         if cfg.experiment.all_seqs_path is not None:
#             logger.info('Loading sequences from all_seqs_path..')
#             all_seqs_df = pd.read_csv(cfg.experiment.all_seqs_path)
#             all_seqs_generated = all_seqs_df.sequence.values.tolist()
#         else:
#             logger.info('Generating sequences by random walk from the base sequence pool..')
#             sequence_cols = cfg.sequence_columns
#             original_seqs = list(set(df_base[sequence_cols].values))
#             num_starting_seqs = len(original_seqs)
#             all_seqs_generated = original_seqs.copy()

#             i_pointer = 0
#             unique_seqs = set()
#             unique_seqs.update(original_seqs)

#             pbar = tqdm(total=max_n_seqs, file=sys.stdout)
#             random_seed = cfg.experiment.random_seed
#             random_obj = random.Random(random_seed)
#             total_iterations = 0
#             while len(all_seqs_generated) < max_n_seqs:
#                 next_seq = all_seqs_generated[i_pointer] # NOTE: This is confusing change naming later
#                 if next_seq not in original_seqs and exploration_method == 'single_mut':
#                     print("WARNING: next_seq not in original_seqs")
#                     break
#                 new_seq = get_next_state(next_seq, task, random_obj)  
#                 pbar.update(1)
#                 if new_seq not in unique_seqs:
#                     all_seqs_generated.append(new_seq)
#                     unique_seqs.add(new_seq)
#                     i_pointer += 1
#                     # Update the progress bar
#                     if cfg.exploration_method == 'single_mut':
#                         i_pointer = i_pointer % num_starting_seqs
#                 total_iterations += 1
#                 if total_iterations >= 2*max_n_seqs:
#                     # go beyond 1 neighbor
#                     #i_pointer = random_obj.randint(0, len(all_seqs_generated)-1)
#                     #
#                     break
#             pbar.close()


#         logger.info("Finished generating sequences by random walk from the base sequence pool..")
#         logger.info("Running predictor on generated sequences..")
#         all_seqs = list(sorted(set(all_seqs_generated)))
#         all_seqs_pt = to_batch_tensor(all_seqs, task, subset=None, device=device)
#         node_scores_init = run_predictor(all_seqs_pt, batch_size=256, predictor=predictor).cpu().numpy()

#         indices_all = np.arange(len(all_seqs))
#         elapsed_time = time.time() - start_time
#         logger.info(f'Finished generation + evaluation in {elapsed_time:.2f} seconds')
#         all_seqs_list = [all_seqs[i] for i in indices_all]
#         logger.info(f'Total of {len(all_seqs_list)} sequences generated')
#         all_seqs_list_orig = deepcopy(all_seqs_list)
#         node_scores_init = node_scores_init[indices_all]
#         encoder = Encoder(alphabet=ALPHABET)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         all_seqs = tensor(encoder.encode(all_seqs_list).to(torch.float).to(device))
        
        
#         logger.info('Creating KNN graph..')
#         start_time = time.time()
#         #set K to be the floor of the square root of the number of sequences + 1
#         fit_KeOps = KNN_KeOps(K=int(np.floor(np.sqrt(len(all_seqs_list)) + 1)), metric='levenshtein')(all_seqs)
#         vals, indices = fit_KeOps(all_seqs)
#         elapsed_time = time.time() - start_time
#         logger.info(f'Finished kNN construction in {elapsed_time:.2f} seconds')
#         vals = 1/vals[:, 1:]
#         indices = indices[:, 1:]
#         non_mutual_knn_graph = csr_matrix((vals.flatten(), indices.flatten(), np.arange(0, len(vals.flatten()) + 1, len(vals[0])))) 
#         mutual_knn_graph = maximum(non_mutual_knn_graph, non_mutual_knn_graph.T)
#         knn_graph = csr_matrix((1/mutual_knn_graph.data, mutual_knn_graph.indices, mutual_knn_graph.indptr))
        
        
#         logger.info('Computing Laplacian..')
#         start_time = time.time()
#         L = laplacian(knn_graph, normed=True).tocsr()
#         gamma = cfg.gamma
#         save_laplacian = cfg.experiment.save_laplacian
#         if save_laplacian:
#             # Construct the file name parameters
#             from scipy.sparse import save_npz
#             params = f'n-{max_n_seqs}_g-{gamma}_seed-{cfg.experiment.random_seed}'
#             graph_dir = os.path.join(data_dir, exploration_method)
#             os.makedirs(graph_dir, exist_ok=True)
            
#             # File paths for Laplacian graphs
#             filename = f'laplacian-{params}.npz'
#             if cfg.experiment.all_seqs_path is not None:
#                 filename = f'laplacian-{params}-from_stored_seqs.npz'
#             laplacian_path = os.path.join(graph_dir, filename)
#             logger.info(f'Saving Laplacian to {laplacian_path}')
#             save_npz(laplacian_path, L)

#     S_init = node_scores_init.copy()
#     gamma = cfg.gamma
#     n = L.shape[0]
#     I = identity(n, format='csr')
#     if cfg.smoothing_method == 'tik':
#         # Tikhonov regularization using the graph Laplacian
#         gamma = cfg.gamma
#         A = I + gamma * L  # L should be your Laplacian matrix
#     elif cfg.smoothing_method == 'L2':
#         # L2 regularization
#         lambda_reg = cfg.gamma  # Your L2 regularization parameter
#         A = I + lambda_reg * I
#     else:
#         raise ValueError(f"Unknown smoothing method: {cfg.experiment.smoothing_method}")

#     Y_opt, info = cg(A, S_init)
#     logger.info('storing results..')
#     df_smoothed = pd.DataFrame({'sequence': all_seqs_list_orig, 'target': Y_opt})

#     now = datetime.now().strftime("%m_%d_%Y_%H_%M") 
#     max_n_seqs = cfg.max_n_seqs
#     gamma = cfg.gamma  # Ensure gamma is defined in your configuration
#     exploration_method = cfg.exploration_method
#     smoothing_method = cfg.smoothing_method
#     params = f'explore-{exploration_method}_smooth-{smoothing_method}_n-{max_n_seqs}_g-{gamma}'
#     results_dir = os.path.join(data_dir, params)

#     # Make sure the necessary directories are created
#     os.makedirs(results_dir, exist_ok=True)

#     # Determine the results file name
#     if cfg.results_file is not None:
#         results_filename = f'{cfg.results_file}-{now}'
#     else:
#         results_filename = f'smoothed_data-{now}'

#     # Construct the full path for the results CSV file
#     results_path = os.path.join(results_dir, f'{results_filename}.csv')

#     # Log the path information and save the CSV file
#     logger.info(f'Writing results to {results_path}')
#     df_smoothed.to_csv(results_path, index=None)

#     # Construct the full path for the configuration YAML file
#     cfg_write_path = os.path.join(results_dir, f'{results_filename}.yaml')

#     # Save the configuration file
#     with open(cfg_write_path, 'w') as f:
#         OmegaConf.save(config=cfg, f=f)

# if __name__ == '__main__':
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#     main()
