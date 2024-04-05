

source env/bin/activate
mkdir -p logs
#used b3 of 0.8 which is fixed throughout the training and runs caspr_adaptive with hparam_1 for 5 times with different seeds.
EXP_DIR="/home/saisurya/Projects/algorithmic-efficiency/caspr_adaptive/caspr_adaptive_submission"
EXP_NAME="submission_$1"
SUBMISSION_PATH="submission_folder/external_tuning/caspr_adaptive/submission.py"
SEARCH_SPACE_PATH="submission_folder/external_tuning/caspr_adaptive/tuning_search_space.json"
WORKLOAD="imagenet_resnet"
DATA_DIR="/mnt/disks/imagenetdata/imagenet/jax"
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
    --num_tuning_trials=1 \
    --eval_period=1000 \
    --max_global_steps=186666  \
    --overwrite > ${LOG_FILE} 2>&1 