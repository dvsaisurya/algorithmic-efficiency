
#used a b2 of 0.8 for lambdas of caspr adaptive but interpolation to casprhm afterwards with adagrad update
EXP_DIR="/home/saisurya/Projects/algorithmic-efficiency/eff_caspr_hm_adaptive"
EXP_NAME="eff_caspr_hm_adaptive_corrected_adagrad_lambeps_1e-2_b2_0.8_interpolation_upto_10k_8_all_hparams"
SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_adaptive_target_setting.py"
SEARCH_SPACE_PATH="prize_qualification_baselines/external_tuning/tuning_search_space.json"
WORKLOAD="ogbg"
DATA_DIR="~/data"
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
    --num_tuning_trials=5