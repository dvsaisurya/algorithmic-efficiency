git clone https://github.com/mlcommons/algorithmic-efficiency.git
cd algorithmic-efficiency
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -e '.[ogbg]'
pip3 install -e '.[wmt]'
pip3 install torchvision
pip install pydub
DATA_DIR=’~/data’
python3 datasets/dataset_setup.py --data_dir $DATA_DIR --ogbg
python3 datasets/dataset_setup.py --data_dir $DATA_DIR --wmt
pip install wandb
pip install jraph
pip install scikit-learn
pip install sacrebleu
pip install sentencepiece