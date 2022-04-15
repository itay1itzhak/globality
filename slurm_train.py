import os
from invoke import run
import itertools
from datetime import datetime
import time
import argparse
import torch  # verify torch is avilable when lanching jobs

timestamp = datetime.now().strftime("%M_%S")


def run_main(args):

    home_dir = args.home_dir
    python_path = args.python_path
    code_dir = args.code_dir
    # all_data_types = ["not_permuted", "permuted", '5gram_permuted']
    # all_data_types = ["permuted", '5gram_permuted']
    # all_data_types = ["ahd_0_3_remove_eos"]
    # all_data_types = ["adspan_after_sm_cone"]
    # all_data_types = ["adspan_cone_remove_eos"]
    # all_data_types = ["adspan_2_local_layers_remove_eos"]
    # all_data_types = ["load_all_attn_remove_eos"]
    # all_data_types = ["load_all_attn_nfreeze_remove_eos"]
    # all_data_types = ['bos_and_eos_at_the_end']
    # all_data_types = ["append_bos_remove_eos"]
    # all_data_types = ["append_bos"]
    # all_data_types = ["double_eos"]
    # all_data_types = ["ah_dropout"]
    all_data_types = args.train_types.split(",")

    train_or_eval = args.train_eval_or_save_attn
    zero_attn_perfix = ""  # "zero_attn_right_to_left_eos_free_"#"zero_attn_right_to_left_eos_free_"#"zero_attn_"
    all_data_sizes = args.train_sizes.split(",")  # ['tiny','small','all']
    all_seeds = args.seeds.split(",")
    all_templates_to_eval = args.templates.split(",")
    all_checkpoints = args.checkpoints.split(
        ","
    )  # [i for i in range(10, 18)] + ["best"]
    all_text_types = args.text_types.split(
        ","
    )  # ['synthetic','semi_natural', 'natural']

    all_save_attn_cond = []
    for text_type in all_text_types:
        for i in all_templates_to_eval:
            if "s_conj" in args.conj_types:
                all_save_attn_cond.extend(
                    [
                        f"compositional_mt/systematicity/s_conj/{text_type}/systematicity_{text_type}_{i}_s1_s2",
                        f"compositional_mt/systematicity/s_conj/{text_type}/systematicity_{text_type}_{i}_s1p_s2",
                        f"compositional_mt/systematicity/s_conj/{text_type}/systematicity_{text_type}_{i}_s3_s2",
                    ]
                )
            if "s_np_vp" in args.conj_types:
                all_save_attn_cond.extend(
                    [
                        f"compositional_mt/systematicity/s_np_vp/{text_type}/systematicity_{text_type}_{i}_np",
                        f"compositional_mt/systematicity/s_np_vp/{text_type}/systematicity_{text_type}_{i}_np_prime",
                        f"compositional_mt/systematicity/s_np_vp/{text_type}/systematicity_{text_type}_{i}_vp_prime",
                    ]
                )

    if train_or_eval != "save_attn":
        all_save_attn_cond = [None]
        all_checkpoints = [None]
        save_attn_matricies = ""
    else:
        save_attn_matricies = f" --save-attn-matricies {home_dir}/attn_weigths/"

    for data_type, data_size, save_attn_cond, checkpoint, seed in list(
        itertools.product(
            all_data_types,
            all_data_sizes,
            all_save_attn_cond,
            all_checkpoints,
            all_seeds,
        )
    ):
        configuration = get_configuration(
            args,
            home_dir,
            python_path,
            code_dir,
            data_type,
            data_size,
            seed,
            train_or_eval,
            save_attn_cond,
            checkpoint,
            zero_attn_perfix,
            save_attn_matricies,
        )
        create_files_and_run(configuration)
        time.sleep(0.1)


##############################################################
def get_configuration(
    args,
    home_dir,
    python_path,
    code_dir,
    data_type,
    data_size,
    seed,
    train_or_eval,
    save_attn_cond=None,
    checkpoint="best",
    zero_attn_perfix="",
    save_attn_matricies="",
):
    conf = dict()
    conf["home_dir"] = home_dir
    conf["python_path"] = python_path
    conf["code_dir"] = code_dir
    conf["checkpoints_dir"] = f"{home_dir}/trained_models/opus/{data_size}/"
    conf["seed"] = seed
    conf["data_size"] = data_size
    conf["train_or_eval"] = train_or_eval
    conf["checkpoint"] = checkpoint
    conf["save_dir"] = data_type + f"_seed_{seed}"

    conf["max-tokens"] = "3584"
    conf["patience"] = "10"
    conf["save_attn_matricies"] = save_attn_matricies

    if data_type == "permuted" or data_type == "2_permuted":
        conf[
            "data_path"
        ] = f"{home_dir}/permuted_data/mt-data/en-nl/opus-taboeta/train.tok.shuf.{data_size}/{data_size}"
    elif data_type == "5gram_permuted" or data_type == "2_5gram_permuted":
        conf[
            "data_path"
        ] = f"{home_dir}/permuted_data/mt-data/en-nl/opus-taboeta/train.5gram.tok.shuf.{data_size}/{data_size}"

    # elif 'not_permuted' in data_type:
    else:
        conf[
            "data_path"
        ] = f"{home_dir}/mt-data/en-nl/opus-taboeta/train.tok.shuf.{data_size}.bpe.60000/{data_size}"
    if "remove_eos" in data_type:
        conf["is_remove_eos"] = "--is-remove-eos"
    else:
        conf["is_remove_eos"] = ""

    if train_or_eval == "eval":  # eval on flores
        # conf['data_path'] = f'home_dir/dieuwkehupkes/mt-data/en-nl/opus-taboeta/valid/flores/data-bin-all'
        conf[
            "data_path"
        ] = f"{home_dir}/dieuwkehupkes/mt-data/en-nl/opus-taboeta/valid/flores/data-bin-{data_size}"
        if data_size == "tiny":
            conf[
                "data_path"
            ] = f"{home_dir}/dieuwkehupkes/mt-data/en-nl/opus-taboeta/valid/flores/data-bin-small"
    elif train_or_eval == "save_attn":
        conf["data_path"] = (
            f"{home_dir}/dieuwkehupkes/nmt/" + save_attn_cond + f"/{data_size}"
        )
        if conf["checkpoint"] == "best":
            cp = "_"
            conf["checkpoint"] = "_best"
        else:
            cp = f"_cp_{conf['checkpoint']}_"
        conf["results_path"] = (
            f"{home_dir}/attn_weigths/{zero_attn_perfix}{data_type}_seed_{seed}{cp}"
            + "_".join(save_attn_cond.split("/")[-6:])
            + f"_{data_size}"
        )
        conf["beam"] = "5"
    elif train_or_eval == "train":
        if args.save_epoch_checkpoints:
            conf["epoch-checkpoints"] = ""
        else:
            conf["epoch-checkpoints"] = "--no-epoch-checkpoints"

    if train_or_eval == "eval" or train_or_eval == "save_attn":  # eval on flores
        conf["nodes"] = "1"
        conf["gpus-per-node"] = "1"
        conf["ntasks-per-node"] = "1"
        conf["time_request"] = "30"
        conf["mem"] = "512G"
    else:
        conf["nodes"] = "2"
        conf["gpus-per-node"] = "8"
        conf["ntasks-per-node"] = "1"
        conf["time_request"] = "4320"
        conf["mem"] = "512G"

    return conf


def get_run_file_text(script_file, conf):
    if conf["train_or_eval"] == "train":
        log_suffix = (
            conf["checkpoints_dir"]
            + conf["save_dir"]
            + "/"
            + conf["train_or_eval"]
            + f"_{timestamp}"
        )
    elif conf["train_or_eval"] == "eval":
        log_suffix = (
            conf["checkpoints_dir"] + conf["save_dir"] + "/" + conf["train_or_eval"]
        )
    elif conf["train_or_eval"] == "save_attn":
        log_suffix = (
            conf["checkpoints_dir"]
            + conf["save_dir"]
            + "/"
            + conf["train_or_eval"]
            + f"_{conf['results_path'].split('/')[-1]}"
        )

    print(f"log_suffix=\n{log_suffix}")

    run_file_text = (
        "#!/bin/sh\n"
        + "#SBATCH --job-name="
        + str(conf["seed"])
        + conf["data_size"][0]
        + conf["save_dir"]
        + "\n"
        + "#SBATCH --output="
        + log_suffix
        + f".out\n"
        + "#SBATCH --error="
        + log_suffix
        + f".err\n"
        + f"#SBATCH --partition=learnlab\n"
        + f"#SBATCH --time={conf['time_request']}\n"
        + f"#SBATCH --nodes={conf['nodes']} \n"
        + f"#SBATCH --mem={conf['mem']} \n"
        + f"#SBATCH --gpus-per-node={conf['gpus-per-node']}\n"
        + f"#SBATCH --ntasks-per-node={conf['ntasks-per-node']}\n"
        + "export WANDB_API_KEY=local-10f30b913d0512cf1a4f362ac6c9ffd2e3ceb574 \n"
        + "srun sh "
        + script_file
        + "\n"
    )

    return run_file_text


def translation_get_script_file_text(conf):
    if conf["train_or_eval"] == "train":
        script_file_text = (
            "#!/bin/bash\n\n"
            + "module load anaconda3/5.0.1\nmodule load cudnn/v8.0.3.33-cuda.11.0\nmodule load cuda/11.0\nmodule load openmpi/4.1.0/cuda.11.0-gcc.9.3.0\nsource activate fairseq-20210318\n"
            + f"{conf['python_path']} {conf['code_dir']}/train.py {conf['data_path']} \\\n"
            + "  --arch transformer_wmt_en_de --share-all-embeddings \\\n"
            + "  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \\\n"
            + "  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \\\n"
            + "  --dropout 0.3 --weight-decay 0.0001 \\\n"
            + "  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\\n"
            + f"  --seed {conf['seed']} --patience {conf['patience']} \\\n"
            + f"  --max-tokens {conf['max-tokens']} --fp16 --update-freq 8 \\\n"
            + "  --save-dir "
            + conf["checkpoints_dir"]
            + conf["save_dir"]
            + "  --eval-bleu \\\n"
            + "  --eval-bleu-args '{\"beam\": 5}' \\\n"
            + "  --eval-bleu-detok moses \\\n"
            + "  --eval-bleu-remove-bpe \\\n"
            + "  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --wandb-project locality \\\n"
            + f"  --distributed-world-size 32 --distributed-port 9218  --validate-interval-updates 5000 --save-interval-updates 20000 {conf['epoch-checkpoints']} {conf['is_remove_eos']} "
        )

    elif conf["train_or_eval"] == "eval":
        script_file_text = (
            "#!/usr/bin/env bash\n"
            + f"  {conf['python_path']} {conf['code_dir']}/fairseq_cli/generate.py \\\n"
            + conf["data_path"]
            + " \\\n"
            + "  --path "
            + conf["checkpoints_dir"]
            + conf["save_dir"]
            + "/checkpoint_best.pt \\\n"
            + f"  --batch-size 128 --beam 5 --remove-bpe --gen-subset test {conf['is_remove_eos']} "
        )  # valid is depended on the folder with flores
    elif conf["train_or_eval"] == "save_attn":
        script_file_text = (
            "#!/usr/bin/env bash\n"
            + f"  {conf['python_path']} {conf['code_dir']}/fairseq_cli/generate.py \\\n"
            + conf["data_path"]
            + " \\\n"
            + "  --path "
            + conf["checkpoints_dir"]
            + conf["save_dir"]
            + f"/checkpoint{conf['checkpoint']}.pt \\\n"
            + f"  --batch-size 1 --beam {conf['beam']} --results-path {conf['results_path']} {conf['is_remove_eos']} {conf['save_attn_matricies']} "
        )

    return script_file_text


def create_files_and_run(conf):
    try:
        os.mkdir(conf["checkpoints_dir"] + conf["save_dir"])
    except OSError as error:
        print(error)

    d = datetime.today()
    d.strftime("%Y-%m-%d")

    run_file = (
        f"{conf['home_dir']}/Slurm_files/run_"
        + conf["save_dir"]
        + "_"
        + str(conf["checkpoint"])
        + "_"
        + conf["data_size"]
        + "_"
        + conf["data_path"].replace("/", "_")
        + "_"
        + conf["train_or_eval"]
    )
    script_file = (
        f"{conf['home_dir']}/Slurm_files/"
        + conf["save_dir"]
        + "_"
        + str(conf["checkpoint"])
        + "_"
        + conf["data_size"]
        + "_"
        + conf["data_path"].replace("/", "_")
        + "_command_"
        + conf["train_or_eval"]
        + ".sh"
    )

    run_file_text = get_run_file_text(script_file, conf)
    script_file_text = translation_get_script_file_text(conf)

    with open(run_file, "w+") as f:
        f.write(run_file_text)

    with open(script_file, "w+") as f:
        f.write(script_file_text)

    time.sleep(0.1)

    ret_value = run("sbatch " + run_file)
    print("return value of run is:")
    print(ret_value)
    print("run_file:", run_file)
    print("script_file:", script_file)
    print("Using data:", conf["data_path"])
    print(conf["train_or_eval"])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--home_dir", type=str, default="/checkpoint/itayitzhak", help=""
    )
    parser.add_argument(
        "--python_path",
        type=str,
        default="/private/home/itayitzhak/.conda/envs/fairseq-20210318/bin/python",
        help="",
    )
    parser.add_argument(
        "--code_dir", type=str, default="/private/home/itayitzhak/globality", help=""
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/checkpoint/itayitzhak/dieuwkehupkes/",
        help="",
    )

    parser.add_argument(
        "--train_types",
        type=str,
        default="remove_eos_not_permuted",
        help="Baselines are remove_eos_not_permuted or not_permuted",
    )
    parser.add_argument(
        "--train_eval_or_save_attn",
        type=str,
        default="eval",
        help="train eval or save_attn",
    )

    parser.add_argument("--train_sizes", type=str, default="all")
    parser.add_argument(
        "--seeds", type=str, default="1,2,3,4,5", help="train eval or save_attn"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="best",
        help="use in eval or save_attn: best or numbers",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="use in eval or save_attn: 1 to 10",
    )
    parser.add_argument(
        "--text_types",
        type=str,
        default="synthetic,semi_natural,natural",
        help="use in eval or save_attn: 1 to 10",
    )
    parser.add_argument("--conj_types", type=str, default="s_conj,s_np_vp")

    parser.add_argument("--zero_attn_perfix", type=str, default="")
    parser.add_argument("--save_epoch_checkpoints", type=bool, default=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    run_main(args)
    # print("Slurm not activated, uncomment in main please")

