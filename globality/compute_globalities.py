from asyncio import FastChildWatcher
import numpy as np
import torch
from tqdm.auto import tqdm
import argparse
import itertools
from pathlib import Path
import submitit
from datetime import datetime
import editdistance

# import globality_scores
# import attention_loading
# import attention_patterns
# import consistency
# import eos_raw_emb
# from globality_scores import *
# from attention_loading import *
# from attention_patterns import *
# from consistency import *
# from eos_raw_emb import *


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
    parser.add_argument("--cps", type=str, default="best")
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


def run_main(args):
    all_data_types = args.data_type.split(",")
    all_data_sizes = args.train_size.split(",")
    all_seeds = args.model_seed.split(",")
    all_text_type = args.text_type.split(",")
    all_templates = args.template.split(",")

    if args.cps == "best":
        all_cps = ["best"]
    else:
        all_cps = [
            i for i in range(int(args.cps.split("-")[0]), int(args.cps.split("-")[1]))
        ]

    all_conds = list(
        zip(args.conj_type.split(","), args.eval_type.split(","))
    )  # [('s_conj','s1_s2')]

    all_models_final_results = get_all_models_final_results(
        args,
        all_data_types,
        all_cps,
        all_data_sizes,
        all_text_type,
        all_conds,
        all_seeds,
        all_templates,
    )
    if all_cps == ["best"]:
        all_cps = ""
    else:
        all_cps = "_cp_" + args.cps
    saving_name = f"results_{args.data_type}{all_cps}_{args.train_size}_{args.text_type}_{args.conj_type}_{args.eval_type}_{args.model_seed}_{args.template}_{args.recompte_avg_attn_mat}_{args.is_with_eos}_{args.consistency_only}_{args.is_fine_grained_gloablities}_{args.zero_out_samples}{args.wrong_compute_type}"
    torch.save(all_models_final_results, f"{args.results_dir}{saving_name}.pt")
    print(f"Saved results in:\n{args.results_dir}{saving_name}.pt")
    print("Done")


def update_samples_info(model_results, args):
    cur_model_pred = model_results["Model_path"] + "/generate-test.txt"
    model_results["num_layers"] = args.num_layers
    model_results["num_heads"] = args.num_heads
    model_results["max_position"] = args.max_position

    model_results["all_samples"] = get_all_samples_dict(cur_model_pred)
    model_results["num_samples"] = len(model_results["all_samples"])
    model_results["lengths_dict"] = get_samples_lengths_dict(
        model_results, args.zero_out_samples
    )
    print(f"lengths_dict:{model_results['lengths_dict']}")


def update_consistency(
    model_results,
    conds,
    template,
    seed,
    trained_data_type,
    cp,
    text_type,
    trained_data_size,
):
    if conds[0] == "s_conj" and conds[1] == "s1_s2":
        (
            (s1p, s3),
            systematicity_trace,
            all_s1_p,
            all_s3,
        ) = compute_systematicity_s_conj_globality(
            template, seed, trained_data_type, cp, text_type, trained_data_size
        )
        # model_results['s_conj_s1p_s2_consist'].append(s1p)
        # model_results['s_conj_s3_s2_consist'].append(s3)
        # model_results['all_sampels_s_conj_s1p_s2_consist'].append(reorder_consistency(all_s1_p, model_results['Model_path']))
        # model_results['all_sampels_s_conj_s3_s2_consist'].append(reorder_consistency(all_s3, model_results['Model_path']))
        # model_results['first_consistency_score'].append(s1p)
        # model_results['second_consistency_score'].append(s3)
        model_results["first_consistency_score"] = s1p
        model_results["second_consistency_score"] = s3
        model_results["all_sampels_first_consistency_score"].append(
            reorder_consistency(all_s1_p, model_results["Model_path"])
        )
        model_results["all_sampels_second_consistency_score"].append(
            reorder_consistency(all_s3, model_results["Model_path"])
        )
        if "zero_attn" in trained_data_type:
            (
                org_s1_s2,
                trace,
                all_org_s1_s2,
            ) = compute_systematicity_s_conj_globality_compare_zero_attn_to_org(
                template, seed, trained_data_type, text_type, trained_data_size
            )
            model_results["org_consistency_score"] = org_s1_s2
        else:
            model_results[
                "org_consistency_score"
            ] = 1.0  # to fit the dataframe, a baseline is always consistent with itself

    elif conds[0] == "s_np_vp" and conds[1] == "np":
        (
            (score_np, score_vp),
            systematicity_trace,
            all_np,
            all_vp,
        ) = compute_systematicity_s_np_vp_globality(
            template, seed, trained_data_type, text_type, trained_data_size
        )
        # model_results['s_np_vp_np_prime_consist'].append(score_np)
        # model_results['s_np_vp_vp_prime_consist'].append(score_vp)
        # model_results['first_consistency_score'].append(score_np)
        # model_results['second_consistency_score'].append(score_vp)
        model_results["first_consistency_score"] = score_np
        model_results["second_consistency_score"] = score_vp
        model_results["org_consistency_score"] = None
    else:
        model_results["first_consistency_score"] = None
        model_results["second_consistency_score"] = None
        model_results["org_consistency_score"] = None


def update_length_results(
    args, length, model_results, trained_data_size, text_type, conds, wrong_compute_type
):
    samples_range = get_samples_range(length, model_results, args.zero_out_samples)
    avg_attn_matrix = get_avg_attn_matrix_load(
        samples_range,
        length,
        model_results,
        model_results["num_samples"],
        model_results["Model_path"],
        trained_data_size,
        conds[0],
        conds[1],
        text_type,
        args.zero_out_samples,
        args.recompte_avg_attn_mat,
    )
    gloablity, wmd = get_model_wmd_gloablity_scores(
        avg_attn_matrix,
        length,
        model_results["num_layers"],
        model_results["num_heads"],
        with_eos=args.is_with_eos,
        wrong_compute_type=wrong_compute_type,
    )
    (
        eos_foucesed_heads,
        self_heads,
        local_heads,
        distance_weighted_median,
        globality_scores,
        percentage_of_EOS,
        percentage_of_self,
        percentage_of_local,
    ) = calc_percentage_of_attn_head_type(
        avg_attn_matrix,
        length,
        model_results["num_layers"],
        model_results["num_heads"],
        with_eos=args.is_with_eos,
        wrong_compute_type=wrong_compute_type,
    )
    # consistencies = get_consistencies_per_sample(systematicity_trace, samples_range, all_samples, template, seed, trained_data_type, text_type, trained_data_size)

    # if conds[0] == 's_conj':
    #     curr_model_final_results['all_sampels_s_conj_s1p_s2_consist'].append(consistencies[0])
    #     curr_model_final_results['all_sampels_s_conj_s3_s2_consist'].append(consistencies[1])
    # elif conds[0] == 's_np_vp':
    #     curr_model_final_results['all_sampels_s_np_vp_np_prime_consist'].append(consistencies[0])
    #     curr_model_final_results['all_sampels_s_np_vp_vp_prime_consist'].append(consistencies[1])

    model_results["eos_foucesed_heads"].append(eos_foucesed_heads)
    model_results["self_heads"].append(self_heads)
    model_results["local_heads"].append(local_heads)
    model_results["distance_weighted_median"].append(distance_weighted_median)
    model_results["globality_scores"].append(globality_scores)
    model_results["percentage_of_EOS"].append(percentage_of_EOS)
    model_results["percentage_of_self"].append(percentage_of_self)
    model_results["percentage_of_local"].append(percentage_of_local)

    return gloablity, wmd


def get_model_final_results(
    args, seed, trained_data_size, trained_data_type, cp, text_type, conds, template
):
    model_results = {
        "eos_foucesed_heads": [],
        "self_heads": [],
        "local_heads": [],
        "distance_weighted_median": [],
        "globality_scores": [],
        "percentage_of_EOS": [],
        "percentage_of_self": [],
        "percentage_of_local": [],
        "model_bleu": [],
        "model_distance_weighted_median": [],
        "model_globality_scores": [],
        "all_sampels_gloablity": [],
        "all_sampels_weighted_median_distance": [],
        "first_consistency_score": [],
        "second_consistency_score": [],
        "all_sampels_first_consistency_score": [],
        "all_sampels_second_consistency_score": [],
        "trained_data_type": trained_data_type,
        "cp": cp,
        "trained_data_size": trained_data_size,
        "conj_type": conds[0],
        "eval_type": conds[1],
        "text_type": text_type,
        "template": template,
        "seed": seed,
    }

    if cp == "best":
        cp = ""
    else:
        cp = f"_cp_{cp}"

    model_results[
        "Model_path"
    ] = f"{args.attn_dir}{trained_data_type}_seed_{seed}{cp}_{args.premuted_prefix}systematicity_{conds[0]}_{text_type}_systematicity_{text_type}_{template}_{conds[1]}{args.premuted_suffix}_{trained_data_size}"
    update_samples_info(model_results, args)
    update_consistency(
        model_results,
        conds,
        template,
        seed,
        trained_data_type,
        cp,
        text_type,
        trained_data_size,
    )

    model_results["model_bleu"].append(
        get_bleu_score(f"{trained_data_type}_seed_{seed}", trained_data_size)
    )

    if args.is_fine_grained_gloablities:  # get all globalities scores
        all_g, all_w = get_all_samples_globality_and_wmd(
            model_results, args.is_with_eos, args.wrong_compute_type
        )
        model_results["all_sampels_gloablity"].append(all_g)
        model_results["all_sampels_weighted_median_distance"].append(all_w)

    if args.consistency_only:
        return model_results

    curr_seed_model_wmd = []
    curr_seed_model_glob = []

    for length in model_results["lengths_dict"].keys():
        gloablity, wmd = update_length_results(
            args,
            length,
            model_results,
            trained_data_size,
            text_type,
            conds,
            args.wrong_compute_type,
        )
        curr_seed_model_wmd.append(wmd)
        curr_seed_model_glob.append(gloablity)

    model_results["model_distance_weighted_median"].append(
        np.average(
            curr_seed_model_wmd, weights=list(model_results["lengths_dict"].values())
        )
    )
    model_results["model_globality_scores"].append(
        np.average(
            curr_seed_model_glob, weights=list(model_results["lengths_dict"].values())
        )
    )

    return model_results


# def get_model_final_results(args, final_results, all_seeds, trained_data_size, trained_data_type, text_type, conds, template, premuted_prefix, premuted_suffix, consistency_only, is_fine_grained_gloablities):
#     return curr_model_final_results


def get_all_models_final_results(
    args,
    all_data_types,
    all_cps,
    all_data_sizes,
    all_text_type,
    all_conds,
    all_seeds,
    all_templates,
):
    final_results = {
        "Model_path": [],
        "eos_foucesed_heads": [],
        "self_heads": [],
        "local_heads": [],
        "distance_weighted_median": [],
        "globality_scores": [],
        "percentage_of_EOS": [],
        "percentage_of_self": [],
        "percentage_of_local": [],
        "num_layers": [],
        "max_position": [],
        "num_heads": [],
        "all_samples": [],
        "num_samples": [],
        "lengths_dict": [],
        "trained_data_type": [],
        "cp": [],
        "trained_data_size": [],
        "conj_type": [],
        "eval_type": [],
        "text_type": [],
        "template": [],
        "seed": [],
        "model_bleu": [],
        "model_distance_weighted_median": [],
        "model_globality_scores": [],
        "all_sampels_gloablity": [],
        "all_sampels_weighted_median_distance": [],
        "first_consistency_score": [],
        "second_consistency_score": [],
        "all_sampels_first_consistency_score": [],
        "all_sampels_second_consistency_score": [],
        "org_consistency_score": [],
    }
    if args.eos_null_hypothesis:
        final_results = {
            "Model_path": [],
            "eos_norms_ratio": [],
            "eos_cosine_ratio": [],
            "random_token_norms_ratio": [],
            "random_token_cosine_ratio": [],
            "trained_data_type": [],
            "cp": [],
            "trained_data_size": [],
            "conj_type": [],
            "eval_type": [],
            "text_type": [],
            "template": [],
            "seed": [],
        }

    all_models = list(
        itertools.product(
            all_data_sizes,
            all_data_types,
            all_cps,
            all_seeds,
            all_text_type,
            all_conds,
            all_templates,
        )
    )

    pb = tqdm(total=len(all_models))
    for (
        trained_data_size,
        trained_data_type,
        cp,
        seed,
        text_type,
        conds,
        template,
    ) in all_models:
        print("=" * 80)
        print(
            f"{trained_data_size}_{cp}_{trained_data_type}_{text_type}_{conds}_{template}"
        )
        print("=" * 80)
        curr_model_final_results = get_model_final_results(
            args,
            seed,
            trained_data_size,
            trained_data_type,
            cp,
            text_type,
            conds,
            template,
        )
        try:
            if not args.eos_null_hypothesis:
                print("+" * 40)
                print(f"Seed={seed}")
                # curr_model_final_results = get_model_final_results(args, seed, trained_data_size, trained_data_type, text_type, conds, template)
            else:
                # print("NOT IMPLEMENTED")

                curr_model_final_results = get_seed_eos_null_results(
                    args,
                    seed,
                    trained_data_size,
                    trained_data_type,
                    text_type,
                    conds,
                    template,
                )
            # model_name_to_save = f"{curr_model_final_results['trained_data_type']}_{trained_data_size}_{text_type}_{conds}_{template}_{args.premuted_prefix}_{args.premuted_suffix}"
            # cur_model_path = f'/checkpoint/itayitzhak/attn_weigths/{trained_data_type}_seed_{seed}_{premuted_prefix}systematicity_{conds[0]}_{text_type}_systematicity_{text_type}_{template}_{conds[1]}{premuted_suffix}_{trained_data_size}'
            for key, value in curr_model_final_results.items():
                final_results[key].append(value)
        except Exception as e:
            print("SKIPPING curr model due to error")
            print(e)
        pb.update(1)
    pb.close()

    return final_results


#################################################### copied to one file so it will work with submitit ###############################
def get_sample_attn_matrix_load(sample_num, model_attn_dir, model_results, is_zero_out):
    """Stack attn_matrix of all layers to a single matrix that belong to one sample"""
    sample_attn_matrix = torch.zeros(
        model_results["num_layers"],
        model_results["num_heads"],
        model_results["max_position"],
        model_results["max_position"],
    )
    layers_matricies = []

    batch_size_times_heads = get_batch_size_times_heads(model_attn_dir, is_zero_out)
    if batch_size_times_heads == model_results["num_heads"]:  # batch size 1
        # print("batch size 1")
        for i in range(model_results["num_layers"]):
            cur_index = sample_num * model_results["num_layers"] + i
            if not is_zero_out:
                layers_matricies.append(
                    torch.load(model_attn_dir + f"/sample_attn_mat_{cur_index}.pt")
                )
            else:
                layers_matricies.append(
                    torch.load(
                        model_attn_dir + f"/sample_attn_mat_zero_out_{cur_index}.pt"
                    )
                )
        sample_mat = torch.stack(layers_matricies, dim=0)
    sample_attn_matrix[
        :, :, : sample_mat.shape[-1], : sample_mat.shape[-1]
    ] = sample_mat

    return sample_attn_matrix


def get_batch_size_times_heads(model_attn_dir, is_zero_out):
    try:
        if not is_zero_out:
            sample_matrix = torch.load(model_attn_dir + f"/sample_attn_mat_0.pt")
        else:
            sample_matrix = torch.load(
                model_attn_dir + f"/sample_attn_mat_zero_out_0.pt"
            )
        batch_size_times_heads = sample_matrix.shape[0]
    except Exception as e:
        list_of_files = glob.glob(f"{model_attn_dir}/avg_attn_mat*")
        latest_avg_matrix = max(list_of_files, key=os.path.getctime)
        # batch_size_times_heads = int(latest_avg_matrix.split('avg_attn_mat_')[1][:2].replace('_','')) # get the length from the avg mat file name
        sample_matrix = torch.load(latest_avg_matrix)
        batch_size_times_heads = sample_matrix.shape[1]
    return batch_size_times_heads


def delete_sample_range_attn_mat(model_attn_dir, samples_range):
    batch_size_times_heads = get_batch_size_times_heads(model_attn_dir)
    for sample_num in range(samples_range[0], samples_range[1] + 1):
        # print(f"Trying to delete: {sample_num} {model_attn_dir} batch_size_times_heads={batch_size_times_heads} NUM_HEADS={NUM_HEADS}")
        if batch_size_times_heads == NUM_HEADS:  # batch size 1
            for i in range(NUM_LAYERS):
                cur_index = sample_num * NUM_LAYERS + i
                # print(f"Trying to delete cur_index: {model_attn_dir}/sample_attn_mat_{cur_index}.pt")
                if os.path.exists(f"{model_attn_dir}/sample_attn_mat_{cur_index}.pt"):
                    # print(f"deleteing... {model_attn_dir}/sample_attn_mat_{cur_index}.pt")
                    os.remove(f"{model_attn_dir}/sample_attn_mat_{cur_index}.pt")


def get_avg_attn_matrix_load(
    samples_range,
    length,
    model_results,
    NUM_SAMPLES,
    model_attn_dir,
    size,
    cond,
    cond_name,
    text_type,
    is_zero_out,
    recompte_avg_attn_mat=False,
):
    range_name = str(samples_range).replace(" ", "_")
    saving_name = (
        model_attn_dir
        + f"/avg_attn_mat_{length}_{range_name}_{size}_{cond}_{cond_name}_{text_type}.pt"
    )
    if not recompte_avg_attn_mat:
        try:
            # print(f"Trying to load: {saving_name}")
            res = torch.load(saving_name)
            # print(f"Using saved avg matrix at {saving_name}")
            # delete_sample_range_attn_mat(model_attn_dir,samples_range)
            return res
        except Exception as e:
            print(f"Exception is:{e}")
            print("No saved avg matrix, calculating and saving")
    print(f"Computing avg matrix for length:{length}...", end=" ")

    sum_attn_weights = torch.zeros(
        (
            model_results["num_layers"],
            model_results["num_heads"],
            model_results["max_position"],
            model_results["max_position"],
        )
    )  # layers X attn_heads X max_position X max_position
    devision_matrix = torch.zeros(
        (
            model_results["num_layers"],
            model_results["num_heads"],
            model_results["max_position"],
            model_results["max_position"],
        )
    )  # layers X attn_heads X max_position X max_position

    # print(f"Sample Range:{samples_range}")
    for sample_num in range(samples_range[0], samples_range[1] + 1):
        sample_attn_matrix = get_sample_attn_matrix_load(
            sample_num, model_attn_dir, model_results, is_zero_out
        )
        sum_attn_weights += sample_attn_matrix
        devision_matrix[sample_attn_matrix != 0] += 1

    devision_matrix[devision_matrix == 0] = 1
    res = sum_attn_weights / devision_matrix

    torch.save(res, saving_name)
    print(f"Saved avg matrix for length: {length}")
    # delete_sample_range_attn_mat(model_attn_dir,samples_range)
    return res


def get_avg_attn_matrix_across_heads(
    samples_heads_list, LENGTH, model_attn_dir, is_zero_out, ignore_eos=False
):
    sum_attn_weights = torch.zeros((LENGTH, LENGTH))
    devision_matrix = torch.zeros((LENGTH, LENGTH))

    for score, sample_num, layer, head in samples_heads_list:
        sample_attn_matrix = get_sample_attn_matrix_load(
            sample_num, model_attn_dir, is_zero_out
        )
        # print(f"sample_attn_matrix.shape:{sample_attn_matrix.shape}")
        sample_attn_matrix = sample_attn_matrix[layer][head][:LENGTH, :LENGTH]
        # print(f"sample_attn_matrix.shape:{sample_attn_matrix.shape}")
        # print(f"sum_attn_weights.shape:{sum_attn_weights.shape}")
        if ignore_eos and is_eos_focused(sample_attn_matrix):
            continue
        sum_attn_weights += sample_attn_matrix
        devision_matrix[sample_attn_matrix != 0] += 1

    print(f"Number of matrices averaged upon: {devision_matrix[0][0]}")

    devision_matrix[devision_matrix == 0] = 1
    return sum_attn_weights / devision_matrix


def get_samples_from_prints():
    all_samples = dict()
    with open(fname, "r") as f:
        sample_num = 0
        line = f.readline()
        while len(line) != 0:  # not EOF
            if "</s>" in line:
                all_samples[sample_num] = line
                # print(str(sample_num)+line)
                sample_num += 1
            line = f.readline()
    return all_samples


def get_all_samples_dict(fname):
    if "attn_weights2.txt" in fname:
        print("getting samples dict from: attn_weights2.txt")
        return get_samples_from_prints()
    print(f"getting samples dict from: {fname}")
    all_samples = dict()
    with open(fname, "r") as f:
        sample_num = 0
        line = f.readline()
        while len(line) != 0:  # not EOF
            if line.startswith("S-"):
                all_samples[sample_num] = line.split(" ")  # [1:]
                # print(str(sample_num)+line)
                sample_num += 1
            line = f.readline()
    return all_samples


def get_sample_len(sample_num, model_results, is_zero_out):
    # model_attn_dir, num_of_samples, num_layers, num_heads
    model_attn_dir = model_results["Model_path"]
    # try:
    #     if not is_zero_out:
    #         sample_matrix = torch.load(model_attn_dir + f'/sample_attn_mat_{sample_num}.pt')
    #     else:
    #         sample_matrix = torch.load(model_attn_dir + f'/sample_attn_mat_zero_out_{sample_num}.pt')
    # except Exception as e: # there's exists a saved lengths dict
    #     saving_name = f'{model_attn_dir}/lengths_dict.pt'
    #     lengths_dict = load_samples_lengths_dict(model_attn_dir, saving_name)

    #     for length in lengths_dict.keys():
    #         loading_name = f'{model_attn_dir}/{length}_range_tuple.pt'
    #         loaded_range = torch.load(loading_name)
    #         if (loaded_range[length][0] <= sample_num) or (sample_num <= loaded_range[length][1]):
    #             return length

    # if sample_matrix.shape[0] == 8: # batch size 1
    index = sample_num * model_results["num_layers"]
    # else: # batch size 128
    #     batch_size = int(torch.load(model_attn_dir + f'/sample_attn_mat_{sample_num}.pt').shape[0]/model_results['num_heads'])
    #     index = int(sample_num / batch_size)*model_results['num_layers']
    if not is_zero_out:
        attn_mat = torch.load(model_attn_dir + f"/sample_attn_mat_{index}.pt")
    else:
        attn_mat = torch.load(model_attn_dir + f"/sample_attn_mat_zero_out_{index}.pt")

    return attn_mat.shape[-1]


def load_samples_lengths_dict(model_attn_dir, saving_name):
    if os.path.exists(saving_name):
        print(f"Loading lengths_dict from: {model_attn_dir}")
        return torch.load(saving_name)
    else:
        print(f"No saved lengths_dict for {model_attn_dir}")
        return None


def get_samples_lengths_dict(model_results, is_zero_out):
    model_attn_dir = model_results["Model_path"]
    saving_name = f"{model_attn_dir}/lengths_dict.pt"
    # samples_legnth = load_samples_lengths_dict(model_attn_dir, saving_name)
    samples_legnth = None
    if samples_legnth == None:
        print(f"Creating new lengths_dict for: {model_attn_dir}")
        samples_legnth = dict()
        for sample_num in range(model_results["num_samples"]):
            cur_len = get_sample_len(sample_num, model_results, is_zero_out)
            if cur_len in samples_legnth:
                samples_legnth[cur_len] += 1
            else:
                samples_legnth[cur_len] = 1

        torch.save(samples_legnth, saving_name)
    return samples_legnth


def get_samples_range(target_length, model_results, is_zero_out):
    # model_results['Model_path'], model_results['num_samples'], model_results['num_layers'],model_results['num_heads']
    model_attn_path = model_results["Model_path"]
    saving_name = f"{model_attn_path}/{target_length}_range_tuple.pt"
    # if os.path.exists(saving_name):
    #     return torch.load(saving_name)[target_length]

    samples_cur_legnth = []
    for sample_num in range(model_results["num_samples"]):
        cur_len = get_sample_len(sample_num, model_results, is_zero_out)
        # print(f"sample_num:{sample_num} | cur_len:{cur_len}")
        if cur_len == target_length:
            samples_cur_legnth.append(sample_num)
    if len(samples_cur_legnth) == 0:
        raise Exception(
            f"No samples found with length {target_length} for model at {model_attn_path}."
        )
    res = {target_length: (samples_cur_legnth[0], samples_cur_legnth[-1])}
    torch.save(res, saving_name)
    return res[target_length]


import numpy as np
import torch
import glob
import os
from globality_scores import *


def is_eos_focused(sample_attn_matrix):
    weight_threshold = 0.20  # 2*1/LENGTH
    majority_threshold = 0.80
    eos_above_threshold = torch.sum(sample_attn_matrix[:, -1] > weight_threshold)
    return (eos_above_threshold / sample_attn_matrix.shape[-1]) > majority_threshold


def get_bleu_score(trained_model, size):
    if "zero_attn" in trained_model:
        return 0.0
    print(
        f"Getting BLEU from: /checkpoint/itayitzhak/trained_models/opus/{size}/{trained_model}/eval*out*"
    )
    list_of_files = glob.glob(
        f"/checkpoint/itayitzhak/trained_models/opus/{size}/{trained_model}/eval*out*"
    )
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, "r") as f:
        for line in f.readlines():
            if "BLEU4" in line:
                return float(line.split("BLEU4")[1][3:8].strip().replace(",", ""))
    return None


def calc_percentage_of_attn_head_type(
    attn_mat,
    LENGTH,
    num_layers,
    num_heads,
    with_eos,
    weight_threshold=0.35,
    majority_threshold=0.9,
    wrong_compute_type="",
):
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
            cur_head = attn_mat[i][j][:LENGTH, :LENGTH]
            if not with_eos:
                cur_head = cur_head[:-1, :-1]

            # EOS-focused
            eos_foucesed_heads.append(np.percentile(cur_head[:, -1], 90))
            eos_above_threshold = torch.sum(cur_head[:, -1] > weight_threshold)
            # eos_foucesed_heads.append(torch.mean(cur_head[:,-1]))
            if (eos_above_threshold / cur_head.shape[-1]) > majority_threshold:
                num_of_eos_foucesed_heads += 1

            # Self-focused
            # print(sorted([x.item() for x in cur_head.diagonal()]))
            self_heads.append(np.percentile(cur_head.diagonal(), 90))
            self_above_threshold = torch.sum(cur_head.diagonal() > weight_threshold)
            if (self_above_threshold / cur_head.shape[-1]) > majority_threshold:
                num_of_self_heads += 1

            # Local-focused
            local_mean = (
                cur_head.diagonal()[1:-1]
                + cur_head.diagonal(1)[1:]
                + cur_head.diagonal(-1)[:-1]
            )  # /3
            pos_first_local_mean = (
                cur_head.diagonal()[0] + cur_head.diagonal(1)[0]
            )  # / 2
            pos_last_local_mean = (
                cur_head.diagonal()[-1] + cur_head.diagonal(-1)[-1]
            )  # / 2
            local_mean = torch.cat(
                [
                    pos_first_local_mean.unsqueeze(0),
                    local_mean,
                    pos_last_local_mean.unsqueeze(0),
                ]
            )

            local_mean -= cur_head.diagonal()  # do not include self attention

            local_heads.append(np.percentile(local_mean, 90))
            local_above_threshold = torch.sum(cur_head.diagonal() > weight_threshold)
            if (local_above_threshold / cur_head.shape[-1]) > majority_threshold:
                num_of_local_heads += 1

            # Weighted median distance & Globality
            distance_weighted_median.append(
                get_mean_distance_weighted_median(cur_head, wrong_compute_type)
            )
            globality_scores.append(
                calc_global_metric(cur_head, wrong_compute_type).item()
            )

    percentage_of_EOS = num_of_eos_foucesed_heads / (num_layers * num_heads)
    percentage_of_self = num_of_self_heads / (num_heads * num_layers)
    percentage_of_local = num_of_local_heads / (num_heads * num_layers)
    print(
        "Percentage of EOS focused heads:  {:.2%}".format(percentage_of_EOS), end=" | "
    )
    print(
        "Percentage of self focused heads: {:.2%}".format(percentage_of_self), end=" | "
    )
    print("Percentage of Local focused heads: {:.2%}".format(percentage_of_local))

    return (
        eos_foucesed_heads,
        self_heads,
        local_heads,
        distance_weighted_median,
        globality_scores,
        percentage_of_EOS,
        percentage_of_self,
        percentage_of_local,
    )


import numpy as np
import torch
import random
import copy
import glob
import os


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_positions_distances_mat(length, negative=False):
    """
    For length 4:
    distance matrix = [[1,2,3,4],[2,1,2,3],[3,2,1,2],[4,3,2,1]]
    """

    distances_matrix = np.zeros((length, length))
    tri = flatten([list(np.arange(i) + 1) for i in range(length - 1, 0, -1)])
    distances_matrix[np.triu_indices(length, 1)] = tri
    if not negative:
        distances_matrix += distances_matrix.T
    else:
        distances_matrix -= distances_matrix.T

    if not negative:
        return torch.tensor(distances_matrix) + 1
    else:
        return torch.tensor(distances_matrix)


def calc_global_metric(attn_mat, wrong_compute_type):
    """
    Calculate a metric to estimate how much a given attntion head is global in thier weights.
    For a given head, the calculation is a mean weighted sum for every token.
    The score for token in position i = sum_{for every token in position j}(a/num_of_tokens)
    
    A score of 1.0 is completely global, a score of 1/num_of_tokens is completely local.
    """

    cur_max_position = attn_mat.shape[-1]  # maximal number of tokens
    distances_matrix = get_positions_distances_mat(cur_max_position).to(
        attn_mat.device
    )  # global weights, the higher the more global

    if wrong_compute_type == "permute":
        distances_matrix = distances_matrix[
            :, torch.randperm(distances_matrix.size()[1])
        ]
    if wrong_compute_type == "ones":
        distances_matrix = torch.ones(distances_matrix.shape).to(attn_mat.device)

    if wrong_compute_type == "one_weight":
        one_weight_output = torch.zeros(cur_max_position).to(attn_mat.device)
        for i in range(cur_max_position):
            one_weight_output[i] = attn_mat[i, np.random.randint(cur_max_position)]
        return one_weight_output.mean(-1)

    # normlize weights to sum to 1.0, in case they don't (in remove EOS)
    norm_attn_mat = attn_mat / attn_mat.sum(dim=1).unsqueeze(1).expand(attn_mat.shape)

    sum_mat = torch.sum(
        distances_matrix * norm_attn_mat, axis=len(norm_attn_mat.shape) - 1
    )
    normalized_weight_distanced = sum_mat / (
        cur_max_position + 1
    )  # normlize score according to length so max is 1

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
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median


def get_mean_distance_weighted_median(attn_mat, wrong_compute_type):
    num_of_tokens = attn_mat.shape[0]
    dists = get_positions_distances_mat(num_of_tokens, negative=True)

    if wrong_compute_type == "permute":
        dists = dists[:, torch.randperm(dists.size()[1])]
    if wrong_compute_type == "ones":
        dists = torch.ones(dists.shape)
    if wrong_compute_type == "one_weight":
        one_weight_output = torch.zeros(num_of_tokens)
        for i in range(num_of_tokens):
            one_weight_output[i] = attn_mat[i, np.random.randint(num_of_tokens)]
        return np.percentile(one_weight_output, 90)

    # sum_of_weighted_medians = 0
    all_weighted_medians = []
    for index_token in range(num_of_tokens):
        cur_weighted_median = weighted_median(
            dists[index_token], attn_mat[index_token] / sum(attn_mat[index_token])
        )  # scale weights to sum to 1.0 if they are not (in remove EOS)
        # sum_of_weighted_medians += abs(cur_weighted_median)
        all_weighted_medians.append(abs(cur_weighted_median))
    return np.percentile(all_weighted_medians, 90)


def get_model_wmd_gloablity_scores(
    attn_mat, LENGTH, NUM_LAYERS, NUM_HEADS, with_eos=True, wrong_compute_type=""
):
    distance_weighted_median = []
    globality_scores = []
    for i in range(NUM_LAYERS):
        for j in range(NUM_HEADS):
            cur_head = attn_mat[i][j][:LENGTH, :LENGTH]
            if not with_eos:
                cur_head = cur_head[:-1, :-1]
            distance_weighted_median.append(
                get_mean_distance_weighted_median(cur_head, wrong_compute_type)
            )
            globality_scores.append(
                calc_global_metric(cur_head, wrong_compute_type).item()
            )

    model_distance_weighted_median = sum(distance_weighted_median) / len(
        distance_weighted_median
    )
    model_globality_scores = sum(globality_scores) / len(globality_scores)
    return model_globality_scores, model_distance_weighted_median


def get_all_samples_globality_and_wmd(
    model_results, with_eos, is_zero_out, wrong_compute_type
):
    all_g, all_w = [], []
    model_attn_dir = model_results["Model_path"]

    for sample_num in range(model_results["num_samples"]):
        sample_attn_matrix = get_sample_attn_matrix_load(
            sample_num, model_attn_dir, is_zero_out
        )
        length = get_sample_len(sample_num, model_results, is_zero_out)
        g, w = get_model_wmd_gloablity_scores(
            sample_attn_matrix,
            length,
            model_results["num_layers"],
            model_results["num_heads"],
            with_eos=with_eos,
            wrong_compute_type=wrong_compute_type,
        )
        all_g.append(g)
        all_w.append(w)
    return all_g, all_w


####################################################### EOS null-hypthesis #######################################
def get_cosine_sim(emb_for_cosine):
    emb_nor = torch.linalg.norm(emb_for_cosine, ord=2, dim=-1)
    emb_nor = emb_nor.unsqueeze(1).expand(emb_for_cosine.size())
    sims = emb_for_cosine / emb_nor
    sims = sims @ sims.T
    return sims


def get_non_diagonal_elements(mat):
    # get non diagonal elements in a matrix
    n = mat.shape[-1]
    return mat.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)


def get_number_of_samples_emb_values(cur_model_path):
    """This functions relies on that there are number_of_samples*2 + 1 files in the folder.
        number_of_samples for norms, number_of_samples for cosine, and 1 for count.txt
    """
    num_of_files = len(
        [
            name
            for name in os.listdir(cur_model_path)
            if os.path.isfile(os.path.join(cur_model_path, name))
        ]
    )
    return num_of_files - 1
    # return  int((num_of_files - 1) / 2) # count.txt file


def load_emb_values(cur_model_path, num_samples):
    emb_values = []
    for i in range(num_samples):
        emb_values.append(torch.load(cur_model_path + f"/sample_emb_raw_{i}.pt"))
    return emb_values


def get_emb_values(emb, values_type):
    num_samples = len(emb)
    all_values = []
    for i in range(num_samples):
        if values_type == "norms":
            all_values.append(torch.norm(emb[i], dim=2))
        elif values_type == "cosine":
            all_layers = []
            num_of_layers = emb[i].shape[0]
            for j in range(num_of_layers):  # for embeddings matrix at each layer
                all_layers.append(get_cosine_sim(emb[i][j]))
            all_values.append(torch.stack(all_layers))

    return all_values


def get_position_index(is_random_token, num_of_tokens):
    if not is_random_token:
        return -1
    else:
        return np.random.randint(num_of_tokens - 2)


def calc_norms_ratio(num_samples, emb_norms, is_random_token=False):
    """at_position = -1 means the ration is computed relativly to the last token, the EOS."""
    all_ratios = []
    for i in range(num_samples):
        num_of_tokens = emb_norms[i].shape[1]
        at_position = get_position_index(is_random_token, num_of_tokens)
        position_norm = emb_norms[i][:, at_position]
        norms_sum = torch.sum(emb_norms[i], dim=1)
        norms_mean_wo_position = (norms_sum - position_norm) / (num_of_tokens - 1)
        all_ratios.append(position_norm / norms_mean_wo_position)

    return torch.stack(all_ratios).mean(dim=0)


def calc_cosine_ratio(num_samples, emb_cosine, is_random_token=False):
    all_ratios = []
    num_layers = emb_cosine[0].shape[0]
    # print(f"emb_cosine[0].shape={emb_cosine[0].shape}")
    for i in range(num_samples):
        all_layers = []
        for j in range(num_layers):
            # print("+"*80)
            # print(f"get_non_diagonal_elements(emb_cosine[i][j]).shape={get_non_diagonal_elements(emb_cosine[i][j]).shape}")
            mean_cosine_per_token = torch.mean(
                get_non_diagonal_elements(torch.abs(emb_cosine[i][j])), dim=1
            )
            # print(mean_cosine_per_token.shape)
            # print((mean_cosine_per_token[-1] / torch.mean(mean_cosine_per_token[:-1])))
            # print((mean_cosine_per_token[-1].item()), end="|")
            # print((torch.mean(mean_cosine_per_token[:-1])).item())
            num_of_tokens = emb_cosine[i][j].shape[0]
            at_position = get_position_index(is_random_token, num_of_tokens)
            # print(f"at_position={at_position}")
            position_mean_cosine = mean_cosine_per_token[at_position]
            # print(f"num_of_tokens={num_of_tokens}")
            # print(f"mean_cosine_per_token={mean_cosine_per_token}")
            # print(f"position_mean_cosine={position_mean_cosine}")

            cosine_sum = torch.sum(mean_cosine_per_token)
            # print(f"cosine_sum={cosine_sum}")
            cosine_mean_wo_position = (cosine_sum - position_mean_cosine) / (
                num_of_tokens - 1
            )
            # print(f"cosine_mean_wo_position={cosine_mean_wo_position}")

            all_layers.append(position_mean_cosine / cosine_mean_wo_position)
            # all_layers.append(torch.cat([mean_cosine_per_token[-1] , torch.mean(mean_cosine_per_token[:-1])]))

        # print(f"torch.stack(all_layers).shape:{torch.stack(all_layers).shape}")
        all_ratios.append(torch.stack(all_layers))
    return torch.stack(all_ratios).mean(dim=0)


def get_seed_eos_null_results(
    args, seed, trained_data_size, trained_data_type, text_type, conds, template
):
    curr_model_final_results = {
        "eos_norms_ratio": [],
        "eos_cosine_ratio": [],
        "random_token_norms_ratio": [],
        "random_token_cosine_ratio": [],
        "trained_data_type": trained_data_type,
        "trained_data_size": trained_data_size,
        "conj_type": conds[0],
        "eval_type": conds[1],
        "text_type": text_type,
        "template": template,
        "seed": seed,
    }

    curr_model_final_results[
        "Model_path"
    ] = f"/checkpoint/itayitzhak/emb_raw/{trained_data_type}_seed_{seed}_{args.premuted_prefix}systematicity_{conds[0]}_{text_type}_systematicity_{text_type}_{template}_{conds[1]}{args.premuted_suffix}_{trained_data_size}"
    num_samples = get_number_of_samples_emb_values(
        curr_model_final_results["Model_path"]
    )
    print(f"num_samples:{num_samples}")
    emb = load_emb_values(curr_model_final_results["Model_path"], num_samples)
    emb_norms = get_emb_values(emb, "norms")
    emb_cosine = get_emb_values(emb, "cosine")
    print(f"emb_norms[0].shape={emb_norms[0].shape}")
    print(f"emb_cosine[0].shape={emb_cosine[0].shape}")
    curr_model_final_results["eos_norms_ratio"] = calc_norms_ratio(
        num_samples, emb_norms
    ).tolist()
    curr_model_final_results["eos_cosine_ratio"] = calc_cosine_ratio(
        num_samples, emb_cosine
    ).tolist()

    curr_model_final_results["random_token_norms_ratio"] = calc_norms_ratio(
        num_samples, emb_norms, is_random_token=True
    ).tolist()
    curr_model_final_results["random_token_cosine_ratio"] = calc_cosine_ratio(
        num_samples, emb_cosine, is_random_token=True
    ).tolist()

    return curr_model_final_results


# def get_model_eos_null_results(all_seeds, trained_data_size, trained_data_type, text_type, conds, template, premuted_prefix, premuted_suffix):
#     curr_model_final_results = {'eos_norms_ratio':[],
#                             'eos_cosine_ratio':[],
#                             'random_token_norms_ratio':[],
#                             'random_token_cosine_ratio':[],
#                            }

#     for seed in all_seeds:
#         cur_model_path = f'/checkpoint/itayitzhak/emb_raw/{trained_data_type}_seed_{seed}_{premuted_prefix}systematicity_{conds[0]}_{text_type}_systematicity_{text_type}_{template}_{conds[1]}{premuted_suffix}_{trained_data_size}'
#         curr_model_final_results = get_seed_eos_null_results(curr_model_final_results.copy(), cur_model_path)
#     return curr_model_final_results


######################################################## Consistency ######################################################


def get_conjunct(sentence):
    """
    Separate sentence into two parts based on conjunction.
    Args:
        - sentence (str)
    Returns:
        - whether the conjunct was found
        - the conjunct
    """
    conjunction = " en "

    # If conjunct in sentence
    if conjunction in sentence:
        conjunct = sentence.split(conjunction)[1:]
        # If the conjunct is shorter than 3 words you've got the wrong one
        if len(sentence.split(conjunction)[0].split()) < 3:
            conjunct = sentence.split(conjunction)[2:]
        conjunct = conjunction.join(conjunct)
        return True, f"MASK {conjunct}"
    return False, f"MASK {sentence}"


def reorder(lines, get_org=False):
    """
    Reorder lines, since Fairseq shuffles them while translating.
    Order based on the index indicated after "D-...", "S-..." or "H-...".
    
    Args:
        - lines: list of str
    Returns:
        - list of str
    """
    sentences = []
    if not get_org:
        line_marker = "D-"
        sent_index = 2
    else:
        line_marker = "S-"
        sent_index = 1
    for line in lines:
        line = line.split("\t")
        if line_marker in line[0]:
            index = int(line[0].split("-")[1])
            sentence = line[sent_index].strip()
            sentences.append((index, sentence))
    _, sentences = zip(*sorted(sentences))
    assert len(sentences) == 500
    return sentences


def compute_systematicity_s_conj_globality(
    template, seed, expirement_data_type, cp, text_type, data_size
):
    """
    Compute systematicity consistency scores for the S -> S CONJ S setup.
    Args:
        - template (int): 1 ... 10
        - data_type (str): natural | semi_natural | synthetic
        - model (str): format of "transformer_DATA_SEED"
    Returns:
        - s1p (float): consistency score for the S1 -> S1' condition
        - s3 (float): consistency score for the S1 -> S3 condition
        - trace (dict): mistakes made by the model
    """
    trace = dict()
    condition_s1p = []
    condition_s3 = []
    # prefix = f"s_conj/{data_type}/systematicity_{data_type}_{template}"
    home_dir = "/checkpoint/itayitzhak/attn_weigths/"
    source_file_prefix = f"/checkpoint/itayitzhak/dieuwkehupkes/nmt/compositional_mt/systematicity/s_conj/{text_type}/systematicity_{text_type}_{template}"

    # English source sentences
    #     with open(f"{prefix}_s1_s2.en", encoding="utf-8") as f:
    #         srcs = f.readlines()
    with open(f"{source_file_prefix}_s1_s2.en", encoding="utf-8") as f:
        srcs = f.readlines()

    # Gather the translation of the regular setup and the two subconditions
    # prefix = f"s_conj/pred_{data_type}/{model}/systematicity_{data_type}_{template}"
    # with open(f"{prefix}_s1_s2.nl", encoding="utf-8") as f:
    try:
        with open(
            f"{home_dir}{expirement_data_type}_seed_{seed}{cp}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1_s2_{data_size}/generate-test.txt",
            encoding="utf-8",
        ) as f:
            pred_s1_s2 = reorder(f.readlines())
    except Exception as e:
        raise Exception(
            f"Could not load pred_s1_s2 at {home_dir}{expirement_data_type}_seed_{seed}{cp}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1_s2_{data_size}/generate-test.txt\n"
            + str(e)
        )

    try:
        # with open(f"{prefix}_s1p_s2.nl", encoding="utf-8") as f:
        with open(
            f"{home_dir}{expirement_data_type}_seed_{seed}{cp}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1p_s2_{data_size}/generate-test.txt",
            encoding="utf-8",
        ) as f:
            pred_s1p_s2 = reorder(f.readlines())
    except Exception as e:
        raise Exception(
            f"Could not load pred_s1p_s2 at {home_dir}{expirement_data_type}_seed_{seed}{cp}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1p_s2_{data_size}/generate-test.txt\n"
            + str(e)
        )

    try:
        # with open(f"{prefix}_s3_s2.nl", encoding="utf-8") as f:
        with open(
            f"{home_dir}{expirement_data_type}_seed_{seed}{cp}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s3_s2_{data_size}/generate-test.txt",
            encoding="utf-8",
        ) as f:
            pred_s3_s2 = reorder(f.readlines())
    except Exception as e:
        raise Exception(
            f"Could not load pred_s3_s2 at {home_dir}{expirement_data_type}_seed_{seed}{cp}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s3_s2_{data_size}/generate-test.txt\n"
            + str(e)
        )

    # print(f"len(srcs)={len(srcs)}", end=" ")
    # print(f"len(pred_s1_s2)={len(pred_s1_s2)}", end=" ")
    # print(f"len(pred_s1p_s2)={len(pred_s1p_s2)}", end=" ")
    # print(f"len(pred_s3_s2)={len(pred_s3_s2)}")
    for src, first, second, third in zip(srcs, pred_s1_s2, pred_s1p_s2, pred_s3_s2):
        # Ensure that the same type of tokenisation is used
        found1, first_conjunct = get_conjunct(first)
        found2, second_conjunct = get_conjunct(second)
        found3, third_conjunct = get_conjunct(third)

        # Don't consider sentences where the second conjunct was not found
        # if not all([found1, found2, found3]):
        #     continue
        # if src == 'The doctor wishes that the mothers sleep definitely , and the child avoids the queen .\n':
        #     print(src)
        if not all([found1, found2]):
            condition_s1p.append(None)
            # condition_s1p.append(False)
            # if found1 and first_conjunct.lower() in second.lower():
            #     condition_s1p.append(True)
            # elif found2 and second_conjunct.lower() in first.lower():
            #     condition_s1p.append(True)
            # elif first.lower() in second.lower() or second.lower() in first.lower():
            #     condition_s1p.append(True)
            # else:
            #     condition_s1p.append(False)
        else:
            condition_s1p.append(first_conjunct == second_conjunct)

        if not all([found1, found3]):
            condition_s3.append(None)
            # condition_s3.append(False)
            # if found1 and third.lower() in first_conjunct.lower():
            #     condition_s3.append(True)
            # elif found3 and third_conjunct.lower() in first.lower():
            #     condition_s3.append(True)
            # elif first.lower() in third.lower() or third.lower() in first.lower():
            #     condition_s3.append(True)
            # else:
            #     condition_s3.append(False)
        else:
            condition_s3.append(first_conjunct == third_conjunct)

        # Collect cases where s2 different for s1' and s3 substitutions
        if first_conjunct not in {second_conjunct, third_conjunct}:
            trace[src.strip()] = (first_conjunct, second_conjunct, third_conjunct)

    np_condition_s1p = np.array(condition_s1p)
    np_condition_s3 = np.array(condition_s3)
    if np_condition_s1p[np_condition_s1p != None].size == 0:
        s1p = 0
    else:
        s1p = np.mean(np_condition_s1p[np_condition_s1p != None])
    if np_condition_s3[np_condition_s3 != None].size == 0:
        s3 = 0
    else:
        s3 = np.mean(np_condition_s3[np_condition_s3 != None])
    return (s1p, s3), trace, condition_s1p, condition_s3


def compute_systematicity_s_conj_globality_compare_zero_attn_to_org(
    template, seed, expirement_data_type, text_type, data_size
):
    trace = dict()
    condition_org_s1_s2 = []

    home_dir = "/checkpoint/itayitzhak/attn_weigths/"
    source_file_prefix = f"/checkpoint/itayitzhak/dieuwkehupkes/nmt/compositional_mt/systematicity/s_conj/{text_type}/systematicity_{text_type}_{template}"

    # English source sentences
    with open(f"{source_file_prefix}_s1_s2.en", encoding="utf-8") as f:
        srcs = f.readlines()

    # Gather the translation of the regular setup and the two subconditions
    try:
        with open(
            f"{home_dir}{expirement_data_type}_seed_{seed}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1_s2_{data_size}/generate-test.txt",
            encoding="utf-8",
        ) as f:
            pred_s1_s2 = reorder(f.readlines())
    except Exception as e:
        raise Exception(
            "Could not load pred_s1_s2 at {{home_dir}{expirement_data_type}_seed_{seed}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1_s2_{data_size}/generate-test.txt}\n"
            + str(e)
        )

    # Gather the original translation of the regular setup and the two subconditions
    try:
        if "remove_eos" in expirement_data_type:
            with open(
                f"{home_dir}remove_eos_not_permuted_seed_{seed}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1_s2_{data_size}/generate-test.txt",
                encoding="utf-8",
            ) as f:
                org_pred_s1_s2 = reorder(f.readlines())
        else:
            with open(
                f"{home_dir}not_permuted_seed_{seed}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1_s2_{data_size}/generate-test.txt",
                encoding="utf-8",
            ) as f:
                org_pred_s1_s2 = reorder(f.readlines())
    except Exception as e:
        raise Exception(
            "Could not load original pred_s1_s2 at {{home_dir}{expirement_data_type}_seed_{seed}_compositional_mt_systematicity_s_conj_{text_type}_systematicity_{text_type}_{template}_s1_s2_{data_size}/generate-test.txt}\n"
            + str(e)
        )

    for src, first, second in zip(srcs, pred_s1_s2, org_pred_s1_s2):
        # Ensure that the same type of tokenisation is used
        found1, first_conjunct = get_conjunct(first)
        found2, second_conjunct = get_conjunct(second)

        # Don't consider sentences where the second conjunct was not found

        if not all([found1, found2]):
            condition_org_s1_s2.append(False)
        else:
            condition_org_s1_s2.append(first_conjunct == second_conjunct)

        # Collect cases where s2 different for s1' and s3 substitutions
        if first_conjunct not in {second_conjunct}:
            trace[src.strip()] = (first_conjunct, second_conjunct)

    org_s1_s2 = np.mean(condition_org_s1_s2)
    return org_s1_s2, trace, condition_org_s1_s2


def compute_systematicity_s_np_vp_globality(
    template, seed, expirement_data_type, text_type, data_size
):
    """
    Compute systematicity consistency scores for the S -> NP VP setup.
    Args:
        - template (int): 1 ... 10
        - data_type (str): natural | semi_natural | synthetic
        - model (str): format of "transformer_DATA_SEED"
    Returns:
        - score_np (float): consistency score for the NP -> NP' condition
        - score_vp (float): consistency score for the VP -> VP' condition
        - trace (dict): mistakes made by the model
    """
    trace = dict()
    condition_np = []
    condition_vp = []
    # prefix = f"s_np_vp/{data_type}/systematicity_{data_type}_{template}"
    # with open(f"{prefix}_np.en", encoding="utf-8") as f:
    home_dir = "/checkpoint/itayitzhak/attn_weigths/"
    source_file_prefix = f"/checkpoint/itayitzhak/dieuwkehupkes/nmt/compositional_mt/systematicity/s_np_vp/{text_type}/systematicity_{text_type}_{template}"
    with open(f"{source_file_prefix}_np.en", encoding="utf-8") as f:
        np_srcs = f.readlines()
    # with open(f"{prefix}_np_prime.en", encoding="utf-8") as f:
    with open(f"{source_file_prefix}_np_prime.en", encoding="utf-8") as f:
        np_srcs_prime = f.readlines()
    if text_type == "synthetic":
        # with open(f"{prefix}_vp_prime.en", encoding="utf-8") as f:
        with open(f"{source_file_prefix}_vp_prime.en", encoding="utf-8") as f:
            vp_srcs_prime = f.readlines()

    # prefix = f"s_np_vp/pred_{data_type}/{model}/systematicity_{data_type}_{template}"
    # with open(f"{prefix}_np.nl", encoding="utf-8") as f:
    with open(
        f"{home_dir}{expirement_data_type}_seed_{seed}_compositional_mt_systematicity_s_np_vp_{text_type}_systematicity_{text_type}_{template}_np_{data_size}/generate-test.txt",
        encoding="utf-8",
    ) as f:
        pred_np = reorder(f.readlines())
    # with open(f"{prefix}_np_prime.nl", encoding="utf-8") as f:
    with open(
        f"{home_dir}{expirement_data_type}_seed_{seed}_compositional_mt_systematicity_s_np_vp_{text_type}_systematicity_{text_type}_{template}_np_prime_{data_size}/generate-test.txt",
        encoding="utf-8",
    ) as f:
        pred_np_prime = reorder(f.readlines())
    if text_type == "synthetic":
        # with open(f"{prefix}_vp_prime.nl", encoding="utf-8") as f:
        with open(
            f"{home_dir}{expirement_data_type}_seed_{seed}_compositional_mt_systematicity_s_np_vp_{text_type}_systematicity_{text_type}_{template}_vp_prime_{data_size}/generate-test.txt",
            encoding="utf-8",
        ) as f:
            pred_vp_prime = reorder(f.readlines())

    for k, (src, src_prime, sent, np_prime) in enumerate(
        zip(np_srcs, np_srcs_prime, pred_np, pred_np_prime)
    ):
        sent = (
            sent.replace(" het ", " de ")
            .replace("Het ", "De ")
            .replace(" dat ", " die ")
        )
        np_prime = (
            np_prime.replace(" het ", " de ")
            .replace("Het ", "De ")
            .replace(" dat ", " die ")
        )

        condition_np.append(editdistance.eval(sent.split(), np_prime.split()) == 1)
        if editdistance.eval(sent.split(), np_prime.split()) != 1:
            trace[(src.strip(), src_prime, "np")] = (sent, np_prime)

        if text_type == "synthetic":
            vp_prime = (
                pred_vp_prime[k]
                .replace(" het ", " de ")
                .replace("Het ", "De ")
                .replace(" dat ", " die ")
            )
            vp_prime = (
                vp_prime.replace(" het ", " de ")
                .replace("Het ", "De ")
                .replace(" dat ", " die ")
            )
            condition_vp.append(editdistance.eval(sent.split(), vp_prime.split()) == 1)
            if editdistance.eval(sent.split(), vp_prime.split()) != 1:
                trace[(np_srcs[k].strip(), vp_srcs_prime[k], "vp")] = (sent, vp_prime)

    # Report results to user in a format that can easily be copied to tables
    score_np = np.mean(condition_np)
    score_vp = None if text_type != "synthetic" else np.mean(condition_vp)
    return (score_np, score_vp), trace, condition_np, condition_vp


def get_consistencies_per_sample(
    systematicity_trace,
    samples_range,
    all_samples,
    template,
    seed,
    trained_data_type,
    text_type,
    trained_data_size,
):
    consistencies = []
    for i, sample in enumerate(all_samples):
        # src_sent = get_src_sent(sample_num, template, seed, trained_data_type, text_type, trained_data_size)
        consistencies.append(systematicity_trace[sample.strip()])
    return consistencies


def reorder_consistency(all_s1_p, cur_model_path):
    cur_model_pred = cur_model_path + "/generate-test.txt"
    ordered_consistency = []

    with open(cur_model_pred, "r") as f:
        lines = f.readlines()

    for sample_num, line in enumerate(lines):
        line = line.split("\t")
        if "S-" in line[0]:
            index = int(line[0].split("-")[1])
            ordered_consistency.append(all_s1_p[index])

    assert len(ordered_consistency) == 500
    return ordered_consistency


def load_pred(model_path, get_org=False):
    try:
        with open(f"{model_path}/generate-test.txt", encoding="utf-8") as f:
            pred = reorder(f.readlines(), get_org)
        return pred
    except Exception as e:
        raise Exception(
            "Could not load pred at {model_path}/generate-test.txt}\n" + str(e)
        )


if __name__ == "__main__":
    args = get_args()
    if args.slurm:
        # Run in Submitit Array
        d = datetime.today()

        if args.cps == "best":
            all_cps = ""
        else:
            all_cps = "_cp_" + args.cps
        saving_name = f"results_{args.data_type}{all_cps}_{args.train_size}_{args.text_type}_{args.conj_type}_{args.eval_type}_{args.model_seed}_{args.template}_{args.recompte_avg_attn_mat}_{args.is_with_eos}_{args.consistency_only}_{args.is_fine_grained_gloablities}_{args.zero_out_samples}"

        exp_dir = (
            Path("/checkpoint/itayitzhak")
            / "projects"
            / "globality"
            / f"{saving_name}"
            # / f"{d.strftime('%Y-%m-%d')}_{args.train_size}_{args.train_size}_{args.text_type}_{args.conj_type}_{args.eval_type}_{args.data_type}_{args.template}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        submitit_logdir = exp_dir / "submitit_logs"

        print("Logs are in:")
        print(submitit_logdir)

        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=180,
            slurm_partition="devlab",
            # slurm_partition="fairlab",
            gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=10,
            slurm_mem="",
        )
        job = executor.submit(run_main, args)
        # print(job.job_id)
        # print(job.exception())
        # print(job.stderr())
    else:
        run_main(args)

