
mkdir -p logs
SETUP_LOG_FILE="logs/setup_log"

bash new_setup.sh >> ${SETUP_LOG_FILE} 2>&1 

source env/bin/activate

#used b3 of 0.8 which is fixed throughout the training and runs caspr_adaptive with hparam_1 for 5 times with different seeds.
EXP_DIR="/home/saisurya/Projects/algorithmic-efficiency/eff_caspr_hm_adaptive_submission"
EXP_NAME="eff_submission_nadamw_ogbg_search_$1"
SUBMISSION_PATH="submission_folder/external_tuning/caspr_adaptive/jax_nadamw_target_setting_0375.py"
SEARCH_SPACE_PATH="caspr_adaptive_ogbg_stack_2_jsons/tuning_search_space_$1.json"
WORKLOAD="ogbg"
DATA_DIR="~/data/ogbg"
RNG_SEED="2"
LOG_FILE="logs/${EXP_NAME}"
python3 submission_runner.py \
    --framework=jax \
    --workload=${WORKLOAD} \
    --experiment_dir=${EXP_DIR}\
    --experiment_name=${EXP_NAME}\
    --submission_path=${SUBMISSION_PATH}\
    --tuning_search_space=${SEARCH_SPACE_PATH}\
    --data_dir=${DATA_DIR} \
    --rng_seed=${RNG_SEED} \
    --num_tuning_trials=10 \
    --eval_period=1000 \
    --max_global_steps=30000  \
    --overwrite >> ${LOG_FILE} 2>&1 &