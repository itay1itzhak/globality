import os
from invoke import run
import itertools
from datetime import datetime
import time

timestamp = datetime.now().strftime("%M_%S")

def run_main():
    # all_data_types = ["not_permuted", "permuted", '5gram_permuted']
    # all_data_sizes = ['small', 'all']
    # all_seeds = [1,2,3,4,5]
    all_data_types = ["not_permuted"]

    # all_data_types = ["remove_eos_not_permuted"]
    # all_data_types = ['bos_and_eos_at_the_end']
    # all_data_types = ["append_bos_remove_eos"]
    # all_data_types = ["append_bos"]
    # all_data_types = ["double_eos"]
    # all_data_types = ["ah_dropout"]

    train_or_eval = 'save_attn' # if 'save_attn', make sure flag is TRUE in code!

    all_templates_to_eval = [1,2,3,4,5,6,7,8,9,10]
    all_data_sizes = ['all']
    all_seeds = [1,2,3,4,5]
    all_text_types = ['synthetic']#['synthetic', 'semi_natural', 'natural']
    
    all_save_attn_cond = []
    for text_type in all_text_types:
        for i in all_templates_to_eval:
            all_save_attn_cond.extend([
                    f"compositional_mt/systematicity/s_conj/{text_type}/systematicity_{text_type}_{i}_s1_s2",
                    f"compositional_mt/systematicity/s_conj/{text_type}/systematicity_{text_type}_{i}_s1p_s2",
                    f"compositional_mt/systematicity/s_conj/{text_type}/systematicity_{text_type}_{i}_s3_s2",
                    f"compositional_mt/systematicity/s_np_vp/{text_type}/systematicity_{text_type}_{i}_np",
                    f"compositional_mt/systematicity/s_np_vp/{text_type}/systematicity_{text_type}_{i}_np_prime",
                    f"compositional_mt/systematicity/s_np_vp/{text_type}/systematicity_{text_type}_{i}_vp_prime"
                    ])


    if train_or_eval != 'save_attn':
        all_save_attn_cond = [None]

    for data_type, data_size, save_attn_cond, seed in list(itertools.product(all_data_types, all_data_sizes, all_save_attn_cond, all_seeds)):
        configuration = get_configuration(data_type,data_size,seed,train_or_eval,save_attn_cond)
        create_files_and_run(configuration)
        time.sleep(0.1)

##############################################################
def get_configuration(data_type,data_size,seed,train_or_eval,save_attn_cond=None):
    conf = dict()
    home_dir = '/checkpoint/itayitzhak/'
    conf['checkpoints_dir'] = home_dir + f'trained_models/opus/{data_size}/'
    conf['seed'] = seed
    conf['data_size'] = data_size
    conf['train_or_eval'] = train_or_eval
    conf['save_dir'] = data_type + f'_seed_{seed}' #'5gram_permuted'#'permuted'# 'not_permuted'
    
    conf['max-tokens'] = '3584'
    
    if data_type == 'permuted' or data_type == '2_permuted':
        conf['data_path'] = home_dir+f'permuted_data/mt-data/en-nl/opus-taboeta/train.tok.shuf.{data_size}/{data_size}'
    elif data_type == '5gram_permuted' or data_type == '2_5gram_permuted':
        conf['data_path'] = home_dir+f'permuted_data/mt-data/en-nl/opus-taboeta/train.5gram.tok.shuf.{data_size}/{data_size}'
    #elif 'not_permuted' in data_type:
    else:
        conf['data_path'] = home_dir+f'mt-data/en-nl/opus-taboeta/train.tok.shuf.{data_size}.bpe.60000/{data_size}'
    
    if train_or_eval == 'eval': # eval on flores
        conf['data_path'] = f'/checkpoint/dieuwkehupkes/mt-data/en-nl/opus-taboeta/valid/flores/data-bin-{data_size}'
        if data_size == 'tiny':
            conf['data_path'] = f'/checkpoint/dieuwkehupkes/mt-data/en-nl/opus-taboeta/valid/flores/data-bin-small'
    elif train_or_eval == 'save_attn':
        conf['data_path'] = "/private/home/dieuwkehupkes/nmt/" + save_attn_cond +f'/{data_size}'
        conf['results_path'] = f"/checkpoint/itayitzhak/attn_weigths/{data_type}_seed_{seed}_" +'_'.join(save_attn_cond.split('/')[-6:]) + f'_{data_size}'
        conf['beam'] = '5'

    if train_or_eval == 'eval' or train_or_eval == 'save_attn': # eval on flores
        conf['nodes'] = '1'
        conf['gpus-per-node'] = '1'
        conf['ntasks-per-node'] = '1'
        conf['time_request'] = '60'
        conf['mem'] = '512G'
    else:
        conf['nodes'] = '2'
        conf['gpus-per-node'] = '8'
        conf['ntasks-per-node'] = '1'
        conf['time_request'] = '4320'
        conf['mem'] = '512G'
    
    return conf


def get_run_file_text(script_file,  conf):
    if conf['train_or_eval'] == 'train':
        log_suffix = conf['checkpoints_dir'] + conf['save_dir'] + "/" + conf['train_or_eval'] + f"_{timestamp}"
    elif conf['train_or_eval'] == 'eval':
        log_suffix = conf['checkpoints_dir'] + conf['save_dir'] + "/" + conf['train_or_eval'] 
    elif conf['train_or_eval'] == 'save_attn':
        log_suffix = conf['checkpoints_dir'] + conf['save_dir'] + "/" + conf['train_or_eval'] + f"_{conf['results_path'].split('/')[-1]}"

    run_file_text = "#!/bin/sh\n" + \
                    "#SBATCH --job-name=" + str(conf['seed']) + conf['data_size'][0] + conf['save_dir'] + "\n" + \
                    "#SBATCH --output=" + log_suffix + f".out\n" + \
                    "#SBATCH --error=" + log_suffix + f".err\n" + \
                    f"#SBATCH --partition=learnlab\n" + \
                    f"#SBATCH --time={conf['time_request']}\n" + \
                    f"#SBATCH --nodes={conf['nodes']} \n" + \
                    f"#SBATCH --mem={conf['mem']} \n" + \
                    f"#SBATCH --gpus-per-node={conf['gpus-per-node']}\n" + \
                    f"#SBATCH --ntasks-per-node={conf['ntasks-per-node']}\n" + \
                    "export WANDB_API_KEY=local-10f30b913d0512cf1a4f362ac6c9ffd2e3ceb574 \n" + \
                    "srun sh " + script_file + "\n"

    return run_file_text

def translation_get_script_file_text(conf):
    if conf['train_or_eval'] == "train":
        script_file_text = "#!/bin/bash\n\n" + \
                           f"/private/home/itayitzhak/.conda/envs/fairseq-20210318/bin/python /private/home/itayitzhak/fairseq/train.py {conf['data_path']} \\\n" + \
                           "  --arch transformer_wmt_en_de --share-all-embeddings \\\n" + \
                           "  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \\\n" + \
                           "  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \\\n" + \
                           "  --dropout 0.3 --weight-decay 0.0001 \\\n" + \
                           "  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\\n" + \
                           f"  --seed {conf['seed']} --patience 10 \\\n" + \
                           f"  --max-tokens {conf['max-tokens']} --fp16 --update-freq 8 \\\n" + \
                           "  --save-dir " + conf['checkpoints_dir'] + conf['save_dir'] + \
                           "  --eval-bleu \\\n" + \
                           "  --eval-bleu-args \'{\"beam\": 5}\' \\\n" + \
                           "  --eval-bleu-detok moses \\\n" + \
                           "  --eval-bleu-remove-bpe \\\n" + \
                           "  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --wandb-project locality \\\n" + \
                           "  --distributed-world-size 32 --distributed-port 9218  --validate-interval-updates 5000 --save-interval-updates 20000 --no-epoch-checkpoints \\\n" 
                        

    elif conf['train_or_eval'] == "eval":
        script_file_text = "#!/usr/bin/env bash\n" + \
                           "  /private/home/itayitzhak/.conda/envs/fairseq-20210318/bin/python /private/home/itayitzhak/fairseq/fairseq_cli/generate.py \\\n" + \
                           conf['data_path'] + " \\\n" + \
                           "  --path " + conf['checkpoints_dir'] + conf['save_dir'] + "/checkpoint_best.pt \\\n" + \
                           "  --batch-size 128 --beam 5 --remove-bpe --gen-subset test" # valid is depended on the folder with flores
    elif conf['train_or_eval'] == "save_attn":
        script_file_text = "#!/usr/bin/env bash\n" + \
                           "  /private/home/itayitzhak/.conda/envs/fairseq-20210318/bin/python /private/home/itayitzhak/fairseq/fairseq_cli/generate.py \\\n" + \
                           conf['data_path'] + " \\\n" + \
                           "  --path " + conf['checkpoints_dir'] + conf['save_dir'] + "/checkpoint_best.pt \\\n" + \
                           f"  --batch-size 1 --beam {conf['beam']} --results-path {conf['results_path']}" 

    return script_file_text



def create_files_and_run(conf):
    try:
        os.mkdir(conf['checkpoints_dir'] + conf['save_dir'])
    except OSError as error:
        print(error)

    run_file = "/checkpoint/itayitzhak/Slurm_files/run_" + conf['save_dir'] + '_' + conf['data_size'] + '_' + conf['data_path'].replace('/','_') + '_' + conf['train_or_eval']
    script_file = "/checkpoint/itayitzhak/Slurm_files/" + conf['save_dir'] + '_' + conf['data_size'] + '_' + conf['data_path'].replace('/','_') + "_command_" + conf['train_or_eval'] + ".sh"
    
    run_file_text = get_run_file_text(script_file, conf)
    script_file_text = translation_get_script_file_text(conf)

    with open(run_file, 'w+') as f:
        f.write(run_file_text)

    with open(script_file, 'w+') as f:
        f.write(script_file_text)
    
    time.sleep(0.1)

    ret_value = run("sbatch " + run_file)
    print("return value of run is:")
    print(ret_value)
    print("run_file:", run_file)
    print("script_file:", script_file)
    print("Using data:", conf['data_path'])
    print(conf['train_or_eval'])


if __name__ == "__main__":
    run_main()
