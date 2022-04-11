import numpy as np
import torch
import seaborn as sns
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import copy
import glob
import os
import pickle
from invoke import run


def get_sample_attn_matrix_load(sample_num, model_attn_dir, model_results):
    '''Stack attn_matrix of all layers to a single matrix that belong to one sample'''
    sample_attn_matrix = torch.zeros(model_results['num_layers'],model_results['num_heads'],model_results['max_position'],model_results['max_position'])
    layers_matricies = []
    
    batch_size_times_heads = get_batch_size_times_heads(model_attn_dir)
    if batch_size_times_heads == model_results['num_heads']: # batch size 1
        #print("batch size 1")
        for i in range(model_results['num_layers']):
            cur_index = sample_num*model_results['num_layers'] + i
            layers_matricies.append(torch.load(model_attn_dir + f'/sample_attn_mat_{cur_index}.pt'))
        sample_mat = torch.stack(layers_matricies, dim=0)
    sample_attn_matrix[:,:,:sample_mat.shape[-1],:sample_mat.shape[-1]] = sample_mat
    
    return sample_attn_matrix

def get_batch_size_times_heads(model_attn_dir):
    try:
        sample_matrix = torch.load(model_attn_dir + f'/sample_attn_mat_0.pt')
        batch_size_times_heads = sample_matrix.shape[0]
    except Exception as e:
        list_of_files = glob.glob(f'{model_attn_dir}/avg_attn_mat*') 
        latest_avg_matrix = max(list_of_files, key=os.path.getctime)
        #batch_size_times_heads = int(latest_avg_matrix.split('avg_attn_mat_')[1][:2].replace('_','')) # get the length from the avg mat file name
        sample_matrix = torch.load(latest_avg_matrix)
        batch_size_times_heads = sample_matrix.shape[1]
    return batch_size_times_heads

def delete_sample_range_attn_mat(model_attn_dir,samples_range):
    batch_size_times_heads = get_batch_size_times_heads(model_attn_dir)
    for sample_num in range(samples_range[0], samples_range[1]+1):
        #print(f"Trying to delete: {sample_num} {model_attn_dir} batch_size_times_heads={batch_size_times_heads} NUM_HEADS={NUM_HEADS}")
        if batch_size_times_heads == NUM_HEADS: # batch size 1
            for i in range(NUM_LAYERS):
                cur_index = sample_num*NUM_LAYERS + i
                #print(f"Trying to delete cur_index: {model_attn_dir}/sample_attn_mat_{cur_index}.pt")
                if os.path.exists(f'{model_attn_dir}/sample_attn_mat_{cur_index}.pt'):
                    #print(f"deleteing... {model_attn_dir}/sample_attn_mat_{cur_index}.pt")
                    os.remove(f'{model_attn_dir}/sample_attn_mat_{cur_index}.pt')
           
        
def get_avg_attn_matrix_load(samples_range, length, model_results, NUM_SAMPLES, model_attn_dir, size, cond, cond_name, text_type,recompte_avg_attn_mat=False):
    range_name = str(samples_range).replace(' ','_')
    saving_name = model_attn_dir + f'/avg_attn_mat_{length}_{range_name}_{size}_{cond}_{cond_name}_{text_type}.pt'
    if not recompte_avg_attn_mat:
        try:
            #print(f"Trying to load: {saving_name}")
            res = torch.load(saving_name)
            # print(f"Using saved avg matrix at {saving_name}")
            # delete_sample_range_attn_mat(model_attn_dir,samples_range)
            return res
        except Exception as e:
            print(f"Exception is:{e}")
            print("No saved avg matrix, calculating and saving")
    print(f"Computing avg matrix for length:{length}...", end=' ')

    sum_attn_weights = torch.zeros((model_results['num_layers'],model_results['num_heads'],model_results['max_position'],model_results['max_position'])) # layers X attn_heads X max_position X max_position
    devision_matrix = torch.zeros((model_results['num_layers'],model_results['num_heads'],model_results['max_position'],model_results['max_position'])) # layers X attn_heads X max_position X max_position

    # print(f"Sample Range:{samples_range}")
    for sample_num in range(samples_range[0], samples_range[1]+1):
        sample_attn_matrix = get_sample_attn_matrix_load(sample_num, model_attn_dir, model_results)
        sum_attn_weights += sample_attn_matrix
        devision_matrix[sample_attn_matrix!=0] += 1

    devision_matrix[devision_matrix==0] = 1
    res = sum_attn_weights/devision_matrix
    
    torch.save(res, saving_name)
    print(f"Saved avg matrix for length: {length}")
    # delete_sample_range_attn_mat(model_attn_dir,samples_range)
    return res


def get_avg_attn_matrix_across_heads(samples_heads_list, LENGTH, model_attn_dir, ignore_eos=False):
    sum_attn_weights = torch.zeros((LENGTH,LENGTH)) 
    devision_matrix = torch.zeros((LENGTH,LENGTH)) 

    for score,sample_num,layer,head in samples_heads_list:
        sample_attn_matrix = get_sample_attn_matrix_load(sample_num, model_attn_dir)
        #print(f"sample_attn_matrix.shape:{sample_attn_matrix.shape}")
        sample_attn_matrix = sample_attn_matrix[layer][head][:LENGTH,:LENGTH]
        #print(f"sample_attn_matrix.shape:{sample_attn_matrix.shape}")
        #print(f"sum_attn_weights.shape:{sum_attn_weights.shape}")
        if ignore_eos and is_eos_focused(sample_attn_matrix):
            continue
        sum_attn_weights += sample_attn_matrix
        devision_matrix[sample_attn_matrix!=0] += 1
    
    print(f"Number of matrices averaged upon: {devision_matrix[0][0]}")
    
    devision_matrix[devision_matrix==0] = 1
    return sum_attn_weights/devision_matrix


def get_samples_from_prints():
    all_samples = dict()
    with open(fname, 'r') as f:
        sample_num = 0
        line = f.readline()
        while len(line) != 0: # not EOF
            if '</s>' in line:
                all_samples[sample_num] = line
                #print(str(sample_num)+line)
                sample_num += 1
            line = f.readline()
    return all_samples
                            
def get_all_samples_dict(fname):
    if "attn_weights2.txt" in fname:
        print("getting samples dict from: attn_weights2.txt")
        return get_samples_from_prints()
    print(f"getting samples dict from: {fname}")
    all_samples = dict()
    with open(fname, 'r') as f:
        sample_num = 0
        line = f.readline()
        while len(line) != 0: # not EOF
            if 'S-' in line:
                all_samples[sample_num] = line.split(' ')#[1:]
                #print(str(sample_num)+line)
                sample_num += 1
            line = f.readline()
    return all_samples

def get_sample_len(sample_num, model_attn_dir, num_of_samples):
    try:
        sample_matrix = torch.load(model_attn_dir + f'/sample_attn_mat_{sample_num}.pt')
    except Exception as e: # there's exists a saved lengths dict
        lengths_dict = get_samples_lengths_dict(num_of_samples, model_attn_dir)
        
        for length in lengths_dict.keys():
            loading_name = f'{model_attn_dir}/{length}_range_tuple.pt'
            loaded_range = torch.load(loading_name)
            if (loaded_range[length][0] <= sample_num) or (sample_num <= loaded_range[length][1]):
                return length
    
    if sample_matrix.shape[0] == 8: # batch size 1
        index = sample_num*NUM_LAYERS
    else: # batch size 128
        batch_size = int(torch.load(model_attn_dir + f'/sample_attn_mat_{sample_num}.pt').shape[0]/NUM_HEADS)
        index = int(sample_num / batch_size)*NUM_LAYERS 
    attn_mat = torch.load(model_attn_dir+f'/sample_attn_mat_{index}.pt')
    return attn_mat.shape[-1]


def get_samples_lengths_dict(num_of_samples, model_attn_dir):
    saving_name = f'{model_attn_dir}/lengths_dict.pt'
    if os.path.exists(saving_name):
        print(f"Loading lengths_dict from: {model_attn_dir}")
        return torch.load(saving_name)
                          
    print(f"Creating new lengths_dict for: {model_attn_dir}")
    samples_legnth = dict()
    for sample_num in range(num_of_samples):
        cur_len = get_sample_len(sample_num, model_attn_dir, num_of_samples)
        if cur_len in samples_legnth:
            samples_legnth[cur_len] += 1
        else:
            samples_legnth[cur_len] = 1
                          
    torch.save(samples_legnth,saving_name)
    return samples_legnth

    
def get_samples_range(target_length, num_of_samples, model_attn_dir):
    saving_name = f'{model_attn_dir}/{target_length}_range_tuple.pt'
    if os.path.exists(saving_name):
        return torch.load(saving_name)[target_length]
    
    samples_cur_legnth = []
    for sample_num in range(num_of_samples):
        cur_len = get_sample_len(sample_num, model_attn_dir, num_of_samples)    
        #print(f"sample_num:{sample_num} | cur_len:{cur_len}")
        if cur_len == target_length:
            samples_cur_legnth.append(sample_num)
    if len(samples_cur_legnth) == 0:
        raise Exception(f'No samples found with length {target_length} for model at {model_attn_dir}.')
    res = {target_length: (samples_cur_legnth[0],samples_cur_legnth[-1])}
    torch.save(res, saving_name)
    return res[target_length]