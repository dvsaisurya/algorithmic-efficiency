

gcloud alpha compute tpus tpu-vm attach-disk tpudemoid \
--zone=us-central1-f \
--disk=imagenetdata \
--mode=read-only

 
sudo mkdir -p /mnt/disks/imagenetdata

sudo mount -o discard,defaults /dev/sdb /mnt/disks/imagenetdata
