#   gcloud compute disks create imagenetdisk \
#     --size 200  \
#     --zone us-central1-f \
#     --type pd-balanced

gcloud alpha compute tpus tpu-vm attach-disk 1tpuv2 \
--zone=us-central1-f \
--disk=imagenetdisk \
--mode=read-write
