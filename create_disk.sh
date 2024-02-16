#   gcloud compute disks create imagenetdisk \
#     --size 200  \
#     --zone us-central1-f \
#     --type pd-balanced

# gcloud alpha compute tpus tpu-vm attach-disk 1tpuv2 \
# --zone=us-central1-f \
# --disk=imagenetdisk \
# --mode=read-write

sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
 
sudo mkdir -p /mnt/disks/imagenetdisk

sudo mount -o discard,defaults /dev/sdb /mnt/disks/imagenetdisk

sudo chmod a+w /mnt/disks/imagenetdisk
