from fairseq_cli import generate
import sys

sys.argv = ['_',
    "/private/home/dieuwkehupkes/nmt/compositional_mt/systematicity/s_conj/synthetic/systematicity_synthetic_8_s1p_s2/small",
    "--path",
    "/checkpoint/itayitzhak/trained_models/opus/small/remove_eos_not_permuted_seed_4/checkpoint_best.pt",
    "--batch-size",
    "1",
    "--beam",
    "5",
    "--results-path",
    "/checkpoint/itayitzhak/attn_weigths/remove_eos_not_permuted_seed_4_compositional_mt_systematicity_s_conj_synthetic_systematicity_synthetic_8_s1p_s2_small"
]
generate.cli_main()
