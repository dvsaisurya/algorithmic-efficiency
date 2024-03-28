# gcloud compute tpus tpu-vm create demo-tpu \
#   --zone=us-central1-f \
#   --accelerator-type=v2-8 \
#   --version=tpu-vm-tf-2.15.0-pjrt \
#   --preemptible
  



# for i in {1..100}; do
#   gcloud alpha compute tpus queued-resources create demo-tpu-$i \
#     --node-id demo-tpu-$i \
#     --zone us-central1-f \
#     --accelerator-type v2-8 \
#     --runtime-version tpu-vm-tf-2.15.0-pjrt \
#     --best-effort
#   # Add a sleep period if necessary to avoid overwhelming quota limits
#   sleep 10 
# done


