

gcloud alpha compute tpus tpu-vm attach-disk tpudemoid \
--zone=us-central1-f \
--disk=librispeech \
--mode=read-only

 
sudo mkdir -p /mnt/disks/librispeech

sudo mount -o discard,defaults /dev/sdc /mnt/disks/librispeech

