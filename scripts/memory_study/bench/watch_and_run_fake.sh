#!/bin/bash
# Wait until GPU 0 is free (two checks a minute apart), then run the fake-PG
# timing configs given as arguments, one after another; record GPU sharing after each.
cd /home/anijain/.claude/jobs/194a3a0e/recover/home/local/torchtitan
free0() { [ "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; }
for i in $(seq 1 120); do
  if free0; then sleep 60; free0 && break; fi
  sleep 60
done
for cfg in "$@"; do
  echo "$(date) start $cfg"
  CUDA_VISIBLE_DEVICES=0 NGPU=512 COMM_BACKEND=fake PYTHONPATH=.:scripts/memory_study:/home/anijain/.claude/jobs/194a3a0e/recover/home/local/dist_moe \
    MODULE=kimi_k3_fake_configs CONFIG=$cfg timeout 3600 ./run_train.sh > /home/anijain/.claude/jobs/194a3a0e/recover/fake_tmp/fake_$cfg.log 2>&1
  echo "$(date) end $cfg exit $? ; gpu0 procs now: $(nvidia-smi -i 0 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
done
