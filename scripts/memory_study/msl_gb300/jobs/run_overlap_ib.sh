# Collective overlap across hosts over InfiniBand (FSDP across NVLink domains):
# all-gather / reduce-scatter / all-to-all over all GPUs of the job, compute alongside.
cd "$(dirname "$0")"; source ./gpu_env.sh
NCCL_MNNVL_ENABLE=0 NCCL_DEBUG=WARN timeout 1500 torchrun --nnodes $NN --nproc-per-node 4 --node-rank $NODE_RANK \
  --master-addr $MASTER --master-port 29641 kimi_k3_overlap_bench.py --sizes-mb 256 1024 \
  --out /tmp/overlap_ib.json > /tmp/overlap_ib.log 2>&1
echo "=== exit $?"
grep -E "MB  blocks|Wrote" /tmp/overlap_ib.log
grep -E "Traceback|Error" /tmp/overlap_ib.log | grep -v "error_handler_fb\|ERROR_HANDLING\|RETHROW\|ScubaLogger" | head -5
