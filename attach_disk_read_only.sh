

gcloud alpha compute tpus tpu-vm attach-disk 6tpuv3 \
--zone=europe-west4-a \
--disk=criteo2 \
--mode=read-only

 
sudo mkdir -p /mnt/disks/criteodata

sudo mount -o discard,defaults /dev/sdb /mnt/disks/criteodata

