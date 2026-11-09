# Overlap of all-gather / reduce-scatter in groups of one GPU per host (an
# expert shard's FSDP replicas, one per NVL domain), across InfiniBand.
cd "$(dirname "$0")"; source ./gpu_env.sh
NCCL_MNNVL_ENABLE=0 NCCL_DEBUG=WARN timeout 1500 torchrun --nnodes $NN --nproc-per-node 4 --node-rank $NODE_RANK \
  --master-addr $MASTER --master-port 29931 kimi_k3_overlap_bench.py --rail --sizes-mb 256 1024 \
  --collectives all_gather reduce_scatter --out /tmp/rov.json > /tmp/rov.log 2>&1
echo "=== exit $?"
grep -E "MB  blocks" /tmp/rov.log
grep -E "Traceback|Error" /tmp/rov.log | grep -v "error_handler_fb\|ERROR_HANDLING\|RETHROW\|ScubaLogger" | head -5
[ -f /tmp/rov.json ] && echo "OVERLAP_JSON $(python -c "import json;print(json.dumps(json.load(open('/tmp/rov.json'))))")"
echo DONE
