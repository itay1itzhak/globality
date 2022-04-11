import numpy as np
import torch
from tqdm.auto import tqdm
import argparse
import itertools
from invoke import run

from globality_scores import *
from attention_loading import *
from attention_patterns import *
from consistency import *
from eos_raw_emb import *


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--slurm", action="store_true", help="use slurm")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/checkpoint/itayitzhak/attn_weigths/results/",
        help="",
    )
    parser.add_argument(
        "--attn_dir", type=str, default="/checkpoint/itayitzhak/attn_weigths/", help=""
    )

    parser.add_argument("--data_type", type=str, default="not_permuted")
    parser.add_argument("--model_seed", type=str, default="1,2")
    parser.add_argument("--test_type", type=str, default="systematicity")
    parser.add_argument("--conj_type", type=str, default="s_conj")
    parser.add_argument("--eval_type", type=str, default="s1_s2")
    parser.add_argument("--text_type", type=str, default="synthetic")
    parser.add_argument("--template", type=str, default="1,2")
    parser.add_argument("--train_size", type=str, default="all,small")
    parser.add_argument(
        "--premuted_prefix", type=str, default="compositional_mt_"
    )  #'permuted_data_' #'compositional_mt_' for non-permuted#
    parser.add_argument("--premuted_suffix", type=str, default="")  # '_3gram'

    parser.add_argument("--cp", type=str, default="best")

    parser.add_argument("--num_heads", type=int, default=8, help="")
    parser.add_argument("--num_layers", type=int, default=6, help="")
    parser.add_argument("--max_position", type=int, default=50, help="")
    parser.add_argument("--seed", type=int, default=42, help="")

    parser.add_argument(
        "--recompte_avg_attn_mat", action="store_true", help="recompte_avg_attn_mat"
    )
    parser.add_argument(
        "--consistency_only", action="store_true", help="consistency_only"
    )
    parser.add_argument(
        "--is_fine_grained_gloablities",
        action="store_true",
        help="is_fine_grained_gloablities",
    )
    parser.add_argument(
        "--eos_null_hypothesis", action="store_true", help="eos_null_hypothesis"
    )
    parser.add_argument("--is_with_eos", action="store_true", help="is_with_eos")
    parser.add_argument("--print_examples", action="store_true", help="use slurm")
    parser.add_argument(
        "--zero_out_samples", action="store_true", help="zero_out_samples"
    )
    parser.add_argument("--wrong_compute_type", type=str, default="")

    args = parser.parse_args()

    return args


def main(args):
    all_data_types = args.data_type.split(",")
    # all_cps = args.cp.split(",")
    all_data_sizes = args.train_size.split(",")
    # all_seeds = args.model_seed.split(',')
    all_text_type = args.text_type.split(",")
    # all_templates = args.template.split(',')
    all_conds = list(zip(args.conj_type.split(","), args.eval_type.split(",")))

    all_models = list(
        itertools.product(all_data_sizes, all_data_types, all_text_type, all_conds)
    )
    pb = tqdm(total=len(all_models))

    for trained_data_size, trained_data_type, text_type, conds in all_models:
        args_string = (
            " --slurm "
            + f" --results_dir {args.results_dir}"
            + f" --attn_dir {args.attn_dir}"
            + f" --data_type {trained_data_type}"
            + f" --cp {args.cp}"
            + f" --model_seed {args.model_seed}"
            + f" --test_type {args.test_type}"
            + f" --conj_type {conds[0]}"
            + f" --eval_type {conds[1]}"
            + f" --text_type {text_type}"
            + f" --template {args.template}"
            + f" --train_size {trained_data_size}"
            + f" --premuted_prefix {args.premuted_prefix}"
            + f" --num_heads {args.num_heads}"
            + f" --num_layers {args.num_layers}"
            + f" --max_position {args.max_position}"
            + f" --seed {args.seed}"
        )

        # f" --premuted_suffix {args.premuted_suffix}" + \

        if args.recompte_avg_attn_mat:
            args_string += " --recompte_avg_attn_mat"
        if args.consistency_only:
            args_string += " --consistency_only"
        if args.is_fine_grained_gloablities:
            args_string += " --is_fine_grained_gloablities"
        if args.eos_null_hypothesis:
            args_string += " --eos_null_hypothesis"
        if args.is_with_eos:
            args_string += " --is_with_eos"
        if args.print_examples:
            args_string += " --print_examples"
        if args.zero_out_samples:
            args_string += " --zero_out_samples"

        if len(args.wrong_compute_type) > 0:
            args_string += f" --wrong_compute_type {args.wrong_compute_type}"

        res = run(
            f"python /private/home/itayitzhak/fairseq/globality/compute_globalities.py {args_string}"
        )
        print(res)
        pb.update(1)
    pb.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
