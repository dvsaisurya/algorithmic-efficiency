

gcloud alpha compute tpus tpu-vm attach-disk tpudemoid \
--zone=us-central1-f \
--disk=fastmridata \
--mode=read-only

 
sudo mkdir -p /mnt/disks/fastmridata

sudo mount -o discard,defaults /dev/sdd /mnt/disks/fastmridata

