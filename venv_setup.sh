pip3 install -e '.[pytorch_cpu]'
pip3 install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
pip3 install -e '.[full]'
pip uninstall jax jaxlib --yes
pip install jax[tpu]==0.4.10 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e '.[wandb]'