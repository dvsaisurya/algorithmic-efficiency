
#used b3 of 0.8 which is fixed throughout the training and runs caspr_adaptive with hparam_1 for 5 times with different seeds.
# EXP_DIR="/home/saisurya/Projects/algorithmic-efficiency/eff_caspr_hm_adaptive"
EXP_DIR="/home/saisurya/Projects/algorithmic-efficiency/eff_caspr_hm_adaptive"
# EXP_NAME="eff_caspr_hm_adaptive_full_matrix_adagrad_all_hparams_v2_3_optimal_hparam_search_b3_lamb_eps"
EXP_NAME="eff_caspr_hm_adaptive_full_matrix_adagrad_all_hparams_v2_6_optimal_hparam_search_b3_lamb_eps"
# SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_adaptive_fast_dist_inv_target_setting.py"
SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_adaptive_full_matrix_dist_inv_target_setting.py"
# SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_adaptive_dist_inv_v2_target_setting.py"
# SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_adaptive_dist_inv_target_setting.py"
# SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_adaptive_target_setting.py"
# SUBMISSION_PATH="prize_qualification_baselines/external_tuning/jax_nadamw_target_setting.py"
# SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_shampoo_dist_inv_target_setting.py"
# SEARCH_SPACE_PATH="prize_qualification_baselines/external_tuning/tuning_search_space.json"
SEARCH_SPACE_PATH="prize_qualification_baselines/external_tuning/tuning_search_space_optimal_hparam_ogbg_search_full_matrix_b3_lamb_eps.json"
# SEARCH_SPACE_PATH="prize_qualification_baselines/external_tuning/tuning_search_space-2.json"
WORKLOAD="ogbg"
DATA_DIR="~/data/ogbg"
RNG_SEED="2"
python3 submission_runner.py \
    --framework=jax \
    --workload=${WORKLOAD} \
    --experiment_dir=${EXP_DIR}\
    --experiment_name=${EXP_NAME}\
    --submission_path=${SUBMISSION_PATH}\
    --tuning_search_space=${SEARCH_SPACE_PATH}\
    --data_dir=${DATA_DIR} \
    --rng_seed=${RNG_SEED} \
    --num_tuning_trials=5 \
    --eval_period=1000 \
    --max_global_steps=30000 