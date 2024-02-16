IMAGENET_TR_URL=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
IMGENET_VAL_URL=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

DATA_DIR="/mnt/disks/imagenetdisk"

python3 datasets/dataset_setup.py --data_dir $DATA_DIR --imagenet --temp_dir $DATA_DIR/tmp --imagenet_train_url $IMAGENET_TR_URL --imagenet_val_url $IMGENET_VAL_URL --framework jax