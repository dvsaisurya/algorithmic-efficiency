# git clone https://github.com/dvsaisurya/algorithmic-efficiency.git
sudo docker builder prune -f
cd docker
sudo docker build -t test_image3 . --build-arg framework=jax
sudo docker run -ti -d  --net=host --ipc=host \
  -v $HOME/data/:/data/ \
  -v $HOME/experiment_runs/:/experiment_runs \
  -v $HOME/experiment_runs/logs:/logs \
  -v $HOME/algorithmic-efficiency:/algorithmic-efficiency \
   --privileged  test_image3 \
  --keep_container_alive true 

