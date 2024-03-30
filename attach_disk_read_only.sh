

gcloud alpha compute tpus tpu-vm attach-disk 1tpuv2 \
--zone=us-central1-f \
--disk=librispeech \
--mode=read-only

 
sudo mkdir -p /mnt/disks/librispeech

sudo mount -o discard,defaults /dev/sdb /mnt/disks/librispeech

