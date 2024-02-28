
DATA_DIR="/home/saisurya/data"
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--temp_dir $DATA_DIR/tmp \
--librispeech

# python3 librispeech_tokenizer.py --train --data_dir=$DATA_DIR/librispeech


# python3 librispeech_preprocess.py --data_dir=$DATA_DIR/librispeech --tokenizer_vocab_path=$DATA_DIR/librispeech/spm_model.vocab