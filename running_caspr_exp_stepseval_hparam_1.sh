
#used a b2 of 0.8 for lambdas of caspr adaptive but interpolation to casprhm afterwards with adagrad update with logging
EXP_DIR="/home/saisurya/Projects/algorithmic-efficiency/eff_caspr_hm_adaptive"
EXP_NAME="eff_caspr_hm_adaptive_corrected_adagrad_lambeps_1e-2_b2_0.8_fixed_with_clipping_with_stepseval_3_hparam_1"
SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_adaptive_target_setting.py"
SEARCH_SPACE_PATH="prize_qualification_baselines/external_tuning/tuning_search_space_hparam_1.json"
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
    --num_tuning_trials=5 \
    --eval_period=1000 \
    --max_global_steps=50000