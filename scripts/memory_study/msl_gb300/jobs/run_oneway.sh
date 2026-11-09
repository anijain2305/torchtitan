# One-way PP-hop overlap over InfiniBand: rank i on host 0 only sends to rank i on
# host 1 while both compute; sender / receiver compute slowdown.
cd "$(dirname "$0")"; source ./gpu_env.sh
MASTER=$(echo $MAST_HPC_TASK_GROUP_HOSTNAMES | cut -d, -f1); NN=$(echo $MAST_HPC_TASK_GROUP_HOSTNAMES | tr ',' '\n' | wc -l)
NCCL_MNNVL_ENABLE=0 NCCL_DEBUG=WARN timeout 1200 torchrun --nnodes $NN --nproc-per-node 4 --node-rank $TW_TASK_ID \
  --master-addr $MASTER --master-port 29631 kimi_k3_overlap_bench.py --one-way --sizes-mb 128 512 1024 \
  --out /tmp/oneway_ib.json > /tmp/oneway.log 2>&1
echo "=== exit $?"
grep -E "MB  blocks|Wrote" /tmp/oneway.log
grep -E "Traceback|Error|error" /tmp/oneway.log | grep -v "error_handler_fb\|ERROR_HANDLING\|RETHROW\|ScubaLogger" | head -10
tail -5 /tmp/oneway.log
[ -f /tmp/oneway_ib.json ] && echo "ONEWAY_JSON $(python -c "import json;print(json.dumps(json.load(open('/tmp/oneway_ib.json'))))")"
