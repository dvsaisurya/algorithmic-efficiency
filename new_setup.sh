sudo apt-get --assume-yes install python3-venv 
python3 -m venv env
source env/bin/activate  
pip install -r requirements.txt
pip uninstall jax jaxlib --yes
pip install jax[tpu]==0.4.10 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html