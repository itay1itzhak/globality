import numpy as np
import torch


def get_cosine_sim(emb_for_cosine):
    emb_nor = torch.linalg.norm(emb_for_cosine, ord=2, dim=-1)
    emb_nor = emb_nor.unsqueeze(1).expand(emb_for_cosine.size())
    sims = emb_for_cosine/emb_nor
    sims = sims @ sims.T
    return sims

def get_non_diagonal_elements(mat):
    # get non diagonal elements in a matrix
    n = mat.shape[-1]
    return mat.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)


def get_number_of_samples_emb_values(cur_model_path):
    '''This functions relies on that there are number_of_samples*2 + 1 files in the folder.
        number_of_samples for norms, number_of_samples for cosine, and 1 for count.txt
    '''
    num_of_files = len([name for name in os.listdir(cur_model_path) if os.path.isfile(os.path.join(cur_model_path, name))])
    return num_of_files - 1
    #return  int((num_of_files - 1) / 2) # count.txt file

def load_emb_values(cur_model_path, num_samples):
    emb_values = []
    for i in range(num_samples):
        emb_values.append(torch.load(cur_model_path + f'/sample_emb_raw_{i}.pt'))
    return emb_values

def get_emb_values(emb,values_type):
    num_samples = len(emb)
    all_values = []
    for i in range(num_samples):
        if values_type == 'norms':
            all_values.append(torch.norm(emb[i],dim=2))
        elif values_type == 'cosine':
            all_layers = []
            num_of_layers = emb[i].shape[0]
            for j in range(num_of_layers): # for embeddings matrix at each layer
                all_layers.append(get_cosine_sim(emb[i][j]))
            all_values.append(torch.stack(all_layers))
    
    return all_values

def get_position_index(is_random_token, num_of_tokens):
    if not is_random_token: 
        return -1
    else:
        return np.random.randint(num_of_tokens-2)


def calc_norms_ratio(num_samples, emb_norms, is_random_token=False):
    '''at_position = -1 means the ration is computed relativly to the last token, the EOS.'''
    all_ratios = []
    for i in range(num_samples):
        num_of_tokens = emb_norms[i].shape[1]
        at_position = get_position_index(is_random_token, num_of_tokens)
        position_norm = emb_norms[i][:,at_position]
        norms_sum = torch.sum(emb_norms[i],dim=1)
        norms_mean_wo_position = (norms_sum - position_norm) / (num_of_tokens - 1)
        all_ratios.append(position_norm / norms_mean_wo_position)
        
    return torch.stack(all_ratios).mean(dim=0)

def calc_cosine_ratio(num_samples, emb_cosine, is_random_token=False):
    all_ratios = []
    num_layers = emb_cosine[0].shape[0]
    #print(f"emb_cosine[0].shape={emb_cosine[0].shape}")
    for i in range(num_samples):
        all_layers = []
        for j in range(num_layers):
            #print("+"*80)
            #print(f"get_non_diagonal_elements(emb_cosine[i][j]).shape={get_non_diagonal_elements(emb_cosine[i][j]).shape}")
            mean_cosine_per_token = torch.mean(get_non_diagonal_elements(torch.abs(emb_cosine[i][j])), dim=1)
            #print(mean_cosine_per_token.shape)
            #print((mean_cosine_per_token[-1] / torch.mean(mean_cosine_per_token[:-1])))
            #print((mean_cosine_per_token[-1].item()), end="|")
            #print((torch.mean(mean_cosine_per_token[:-1])).item())
            num_of_tokens = emb_cosine[i][j].shape[0] 
            at_position = get_position_index(is_random_token, num_of_tokens)
            #print(f"at_position={at_position}")
            position_mean_cosine = mean_cosine_per_token[at_position]
            #print(f"num_of_tokens={num_of_tokens}")
            #print(f"mean_cosine_per_token={mean_cosine_per_token}")
            #print(f"position_mean_cosine={position_mean_cosine}")

            cosine_sum = torch.sum(mean_cosine_per_token)
            #print(f"cosine_sum={cosine_sum}")
            cosine_mean_wo_position = (cosine_sum - position_mean_cosine) / (num_of_tokens - 1)
            #print(f"cosine_mean_wo_position={cosine_mean_wo_position}")
            
            all_layers.append(position_mean_cosine / cosine_mean_wo_position)
            #all_layers.append(torch.cat([mean_cosine_per_token[-1] , torch.mean(mean_cosine_per_token[:-1])]))

        #print(f"torch.stack(all_layers).shape:{torch.stack(all_layers).shape}")
        all_ratios.append(torch.stack(all_layers))
    return torch.stack(all_ratios).mean(dim=0)
    #return calc_norms_ratio(cur_model_path, num_samples, emb_cosine)
    
# bla_norms = torch.tensor([[1.0 ,1 ,1], [1.0 ,3 ,700]])
# bla_cosine = torch.tensor([[[0.1 ,0.9 ,0.1],[0.2, 0.1, 0.1],[0.3, 0.700, 0.1]], [[1.0 ,1 ,1],[1, 1, 1],[3, 700, 1]]])
# #print(bla_norms.shape)
# print(bla_cosine.shape)
# num_samples = 1
# print(calc_norms_ratio(num_samples, [bla_norms], is_random_token=False))
# print(calc_cosine_ratio(num_samples, [bla_cosine], is_random_token=False))

# print(calc_norms_ratio(num_samples, [bla_norms], is_random_token=True))
# print(calc_cosine_ratio(num_samples, [bla_cosine], is_random_token=True))

def get_seed_eos_null_results(curr_model_final_results, cur_model_path):
    num_samples = get_number_of_samples_emb_values(cur_model_path)
    print(f"num_samples:{num_samples}")
    emb = load_emb_values(cur_model_path, num_samples)
    emb_norms = get_emb_values(emb,'norms')
    emb_cosine = get_emb_values(emb,'cosine')
    print(f"emb_norms[0].shape={emb_norms[0].shape}")
    print(f"emb_cosine[0].shape={emb_cosine[0].shape}")
    curr_model_final_results['eos_norms_ratio'] = calc_norms_ratio(num_samples, emb_norms)
    curr_model_final_results['eos_cosine_ratio'] = calc_cosine_ratio(num_samples, emb_cosine)
    
    curr_model_final_results['random_token_norms_ratio'] = calc_norms_ratio(num_samples, emb_norms, is_random_token=True)
    curr_model_final_results['random_token_cosine_ratio'] = calc_cosine_ratio(num_samples, emb_cosine, is_random_token=True)
    
    return curr_model_final_results

def get_model_eos_null_results(args, all_seeds, trained_data_size, trained_data_type, text_type, conds, template):
    curr_model_final_results = {'eos_norms_ratio':[],
                            'eos_cosine_ratio':[],
                            'random_token_norms_ratio':[],
                            'random_token_cosine_ratio':[],
                           }

    size = trained_data_size
    for seed in all_seeds:
        cur_model_path = f'/checkpoint/itayitzhak/emb_raw/{trained_data_type}_seed_{seed}_{args.premuted_prefix}systematicity_{conds[0]}_{text_type}_systematicity_{text_type}_{template}_{conds[1]}{args.premuted_suffix}_{trained_data_size}'
        curr_model_final_results = get_seed_eos_null_results(curr_model_final_results.copy(), cur_model_path)
    return curr_model_final_results


def run_main():
    curr_model_final_results = dict()
    curr_model_final_results = get_seed_eos_null_results(curr_model_final_results, '/checkpoint/itayitzhak/emb_raw/not_permuted_seed_1_permuted_data_systematicity_s_conj_synthetic_systematicity_synthetic_1_s1_s2_small')
    for key, value in curr_model_final_results.items():
        print(key)
        print(value)



if __name__ == "__main__":
    run_main()