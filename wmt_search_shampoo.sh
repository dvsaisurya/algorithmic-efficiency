
bash venv_setup.sh

source env/bin/activate

python3 datasets/dataset_setup.py --data_dir "~/data" --wmt
mkdir -p logs
#used b3 of 0.8 which is fixed throughout the training and runs caspr_adaptive with hparam_1 for 5 times with different seeds.
EXP_DIR="/home/saisurya/Projects/algorithmic-efficiency/eff_caspr_hm_adaptive"
EXP_NAME="eff_shampoo_wmt_search_0"
SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_shampoo_dist_inv_target_setting.py"
SEARCH_SPACE_PATH="tuning_search_space_caspr_adaptive_full_matrix_ogbg_0.json"
WORKLOAD="wmt"
DATA_DIR="~/data/wmt"
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
    --num_tuning_trials=5 \
    --eval_period=1000 \
    --max_global_steps=50000  \
    --overwrite > ${LOG_FILE} 2>&1 &