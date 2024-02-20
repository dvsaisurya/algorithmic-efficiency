  gcloud compute disks create imagenetdata \
    --size 500  \
    --zone us-central1-f \
    --type pd-balanced

gcloud alpha compute tpus tpu-vm attach-disk 1tpuv2 \
--zone=us-central1-f \
--disk=imagenetdata \
--mode=read-write

sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
 
sudo mkdir -p /mnt/disks/imagenetdata

sudo mount -o discard,defaults /dev/sdb /mnt/disks/imagenetdata

sudo chmod a+w /mnt/disks/imagenetdata
