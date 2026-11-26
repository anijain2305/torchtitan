#!/bin/bash
# usage: worker.sh GPU ; pops jobs from queue.txt ("name|offload(0/1)|ENV=V ENV=V ...") under a lock
G=$1; D=/tmp/claude-anijain/sweep; cd /home/anijain/local/torchtitan
NUMA=""; [ "$G" = 2 ] || [ "$G" = 3 ] && NUMA="numactl --preferred=1"
while true; do
  line=$(flock $D/queue.lock bash -c "head -1 $D/queue.txt; sed -i '1d' $D/queue.txt")
  [ -z "$line" ] && break
  name=${line%%|*}; rest=${line#*|}; off=${rest%%|*}; envs=${rest#*|}
  [ "$off" = 1 ] && exec 9>$D/offload.lock && flock 9
  echo "$(date +%H:%M) start $name gpu$G" >> $D/progress.txt
  env $envs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NGPU=512 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=$G \
    PYTHONPATH=.:scripts/memory_study:/home/anijain/local/dist_moe \
    timeout 5400 $NUMA /home/anijain/.conda/envs/kimi_study/bin/torchrun --standalone --nproc-per-node=1 \
    --master-port=$((31000 + G * 37 + RANDOM % 30)) -m torchtitan.train --module kimi_k3_remat_regions \
    --config remat_generic --comm-backend fake > $D/$name.log 2>&1
  rc=$?
  [ "$off" = 1 ] && flock -u 9 && exec 9>&-
  echo "$(date +%H:%M) done $name rc=$rc $(grep -oE 'memory: [0-9.]+GiB' $D/$name.log | tail -1) $(grep -m1 -oE 'OutOfMemoryError' $D/$name.log)" >> $D/progress.txt
done
echo "$(date +%H:%M) worker $G exit" >> $D/progress.txt
