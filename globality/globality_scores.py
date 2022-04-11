import numpy as np
import torch
import random
import copy
import glob
import os



def flatten(t):
    return [item for sublist in t for item in sublist]

def get_positions_distances_mat(length, negative=False):
    '''
    For length 4:
    distance matrix = [[1,2,3,4],[2,1,2,3],[3,2,1,2],[4,3,2,1]]
    '''
    
    distances_matrix = np.zeros((length,length))
    tri = flatten([list(np.arange(i)+1) for i in range(length-1,0,-1)])
    distances_matrix[np.triu_indices(length,1)] = tri
    if not negative:
        distances_matrix += distances_matrix.T
    else:
        distances_matrix -= distances_matrix.T
    
    if not negative: 
        return torch.tensor(distances_matrix)# + 1
    else:
        return torch.tensor(distances_matrix)
    
def calc_global_metric(attn_mat):
    '''
    Calculate a metric to estimate how much a given attntion head is global in thier weights.
    For a given head, the calculation is a mean weighted sum for every token.
    The score for token in position i = sum_{for every token in position j}(a/num_of_tokens)
    
    A score of 1.0 is completely global, a score of 1/num_of_tokens is completely local.
    '''
    
    cur_max_position = attn_mat.shape[-1] # maximal number of tokens
    distances_matrix = get_positions_distances_mat(cur_max_position).to(attn_mat.device) # global weights, the higher the more global
    
    # normlize weights to sum to 1.0, in case they don't (in remove EOS)
    norm_attn_mat = attn_mat / attn_mat.sum(dim=1).unsqueeze(1).expand(attn_mat.shape)
    
    sum_mat = torch.sum(distances_matrix*norm_attn_mat,axis=len(norm_attn_mat.shape)-1)
    normalized_weight_distanced = sum_mat / (cur_max_position + 1) # normlize score according to length so max is 1
    
    return normalized_weight_distanced.mean(-1)
    
def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    
    returns the weighted median in data according to weights (float)
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median
    
def get_mean_distance_weighted_median(attn_mat):
    num_of_tokens = attn_mat.shape[0]
    dists = get_positions_distances_mat(num_of_tokens,negative=True)
    #sum_of_weighted_medians = 0
    all_weighted_medians = []
    for index_token in range(num_of_tokens):
        cur_weighted_median = weighted_median(dists[index_token], attn_mat[index_token]/sum(attn_mat[index_token])) # scale weights to sum to 1.0 if they are not (in remove EOS)
        # sum_of_weighted_medians += abs(cur_weighted_median)
        all_weighted_medians.append(abs(cur_weighted_median))
    return np.percentile(all_weighted_medians, 90)

def get_model_wmd_gloablity_scores(attn_mat, LENGTH, NUM_LAYERS, NUM_HEADS, with_eos=True):
    distance_weighted_median = []
    globality_scores = []
    for i in range(NUM_LAYERS):
        for j in range(NUM_HEADS):
            cur_head = attn_mat[i][j][:LENGTH,:LENGTH]
            if not with_eos:
                cur_head = cur_head[:-1,:-1]
            distance_weighted_median.append(get_mean_distance_weighted_median(cur_head))
            globality_scores.append(calc_global_metric(cur_head).item())
    
    model_distance_weighted_median = sum(distance_weighted_median)/len(distance_weighted_median)
    model_globality_scores = sum(globality_scores)/len(globality_scores)
    return model_globality_scores, model_distance_weighted_median
    

def get_all_samples_globality_and_wmd(model_attn_dir, NUM_SAMPLES, NUM_LAYERS, NUM_HEADS, lengths_dict, with_eos):
    all_g, all_w = [], []
    
    for sample_num in range(NUM_SAMPLES):
        sample_attn_matrix = get_sample_attn_matrix_load(sample_num, model_attn_dir)
        length = get_sample_len(sample_num, model_attn_dir, NUM_SAMPLES)
        g, w = get_model_wmd_gloablity_scores(sample_attn_matrix, length, NUM_LAYERS, NUM_HEADS, with_eos=with_eos)  
        all_g.append(g)
        all_w.append(w)
    return all_g, all_w