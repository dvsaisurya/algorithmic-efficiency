

source env/bin/activate
mkdir -p logs
#used b3 of 0.8 which is fixed throughout the training and runs caspr_adaptive with hparam_1 for 5 times with different seeds.
EXP_DIR="/home/saisurya/Projects/algorithmic-efficiency/eff_caspr_hm_adaptive/eff_caspr_hm_adaptive_full_matrix_imagenet_resnet_search"
EXP_NAME="eff_caspr_hm_adaptive_full_matrix_imagenet_resnet_search_trial_$1"
SUBMISSION_PATH="prize_qualification_baselines/external_tuning/efficient_caspr_adaptive_full_matrix_dist_inv_target_setting_imagenet_resnet.py"
SEARCH_SPACE_PATH="tuning_search_space_caspr_adaptive_full_matrix_imagenet_indiv_$1.json"
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
    --max_global_steps=100000  \
    --overwrite > ${LOG_FILE} 2>&1 