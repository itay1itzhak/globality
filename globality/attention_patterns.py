import numpy as np
import torch
import glob
import os
from globality_scores import *

def is_eos_focused(sample_attn_matrix):
    weight_threshold = 0.20 # 2*1/LENGTH
    majority_threshold = 0.80
    eos_above_threshold = torch.sum(sample_attn_matrix[:,-1] > weight_threshold)
    return (eos_above_threshold / sample_attn_matrix.shape[-1]) > majority_threshold


def get_bleu_score(trained_model, size):
    print(f'Getting BLEU from: /checkpoint/itayitzhak/trained_models/opus/{size}/{trained_model}/eval*out*')
    list_of_files = glob.glob(f'/checkpoint/itayitzhak/trained_models/opus/{size}/{trained_model}/eval*out*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        for line in f.readlines():
            if 'BLEU4' in line:
                return float(line.split('BLEU4')[1][3:8].strip().replace(',',''))
    return None


def calc_percentage_of_attn_head_type(attn_mat, LENGTH, num_layers, num_heads, with_eos, weight_threshold=0.35, majority_threshold=0.9):
#     print(f"Attention Weight threshold is: {weight_threshold:.2f}.")
#     print(f"Majority threshold is: {majority_threshold:.1f}.")
    num_of_eos_foucesed_heads = 0
    num_of_local_heads = 0
    num_of_self_heads = 0
    
    eos_foucesed_heads = []
    self_heads = []
    local_heads = []
    distance_weighted_median = []
    globality_scores = []
    
    for i in range(num_layers):
        for j in range(num_heads):
            cur_head = attn_mat[i][j][:LENGTH,:LENGTH]
            if not with_eos:
                cur_head = cur_head[:-1,:-1]
            
            # EOS-focused
            eos_foucesed_heads.append(np.percentile(cur_head[:,-1], 90))
            eos_above_threshold = torch.sum(cur_head[:,-1] > weight_threshold)
            #eos_foucesed_heads.append(torch.mean(cur_head[:,-1]))
            if (eos_above_threshold / cur_head.shape[-1]) > majority_threshold:
                num_of_eos_foucesed_heads += 1
                
            # Self-focused
            #print(sorted([x.item() for x in cur_head.diagonal()]))
            self_heads.append(np.percentile(cur_head.diagonal(), 90))
            self_above_threshold = torch.sum(cur_head.diagonal() > weight_threshold)
            if (self_above_threshold / cur_head.shape[-1]) > majority_threshold:
                num_of_self_heads += 1
                
            # Local-focused
            local_mean = (cur_head.diagonal()[1:-1] + cur_head.diagonal(1)[1:] + cur_head.diagonal(-1)[:-1])#/3
            pos_first_local_mean = (cur_head.diagonal()[0] + cur_head.diagonal(1)[0]) #/ 2
            pos_last_local_mean = (cur_head.diagonal()[-1] + cur_head.diagonal(-1)[-1]) #/ 2
            local_mean = torch.cat([pos_first_local_mean.unsqueeze(0),local_mean,pos_last_local_mean.unsqueeze(0)])
            
            local_mean -= cur_head.diagonal() # do not include self attention
            
            local_heads.append(np.percentile(local_mean, 90))
            local_above_threshold = torch.sum(cur_head.diagonal() > weight_threshold)
            if (local_above_threshold / cur_head.shape[-1]) > majority_threshold:
                num_of_local_heads += 1
            
            # Weighted median distance & Globality
            distance_weighted_median.append(get_mean_distance_weighted_median(cur_head))
            globality_scores.append(calc_global_metric(cur_head).item())

    print("Percentage of EOS focused heads:  {:.2%}".format(num_of_eos_foucesed_heads/(num_layers*num_heads)), end=' | ')
    print("Percentage of self focused heads: {:.2%}".format(num_of_self_heads/(num_heads*num_layers)), end=' | ')
    print("Percentage of Local focused heads: {:.2%}".format(num_of_local_heads/(num_heads*num_layers)))
    
    return eos_foucesed_heads, self_heads, local_heads, distance_weighted_median, globality_scores