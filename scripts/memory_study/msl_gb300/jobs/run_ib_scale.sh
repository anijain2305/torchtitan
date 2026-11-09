# FSDP- and PP-style traffic across InfiniBand at this job's size: 4 GPUs per
# host on NVLink, hosts over InfiniBand only (multi-host NVLink off). All-gather
# and reduce-scatter bandwidth and overlap over every GPU; send/recv bandwidth;
# one-way hop overlap (rank i -> rank i + world / 2).
cd "$(dirname "$0")"; source ./gpu_env.sh
TR="torchrun --nnodes $NN --nproc-per-node 4 --node-rank $NODE_RANK --master-addr $MASTER"
export NCCL_MNNVL_ENABLE=0 NCCL_DEBUG=WARN
BW_TAG=ib_${NN}hosts BW_OPS=allgather,reducescatter,sendrecv timeout 1200 $TR --master-port 29801 coll_bw.py 2>&1 | grep "^BW "
timeout 1800 $TR --master-port 29802 kimi_k3_overlap_bench.py --sizes-mb 256 1024 \
  --collectives all_gather reduce_scatter --out /tmp/ov.json > /tmp/ov.log 2>&1
echo "=== overlap exit $?"
grep -E "MB  blocks" /tmp/ov.log
grep -E "Traceback|Error" /tmp/ov.log | grep -v "error_handler_fb\|ERROR_HANDLING\|RETHROW\|ScubaLogger" | head -5
[ -f /tmp/ov.json ] && echo "OVERLAP_JSON $(python -c "import json;print(json.dumps(json.load(open('/tmp/ov.json'))))")"
timeout 1200 $TR --master-port 29803 kimi_k3_overlap_bench.py --one-way --sizes-mb 1024 --out /tmp/ow.json > /tmp/ow.log 2>&1
echo "=== oneway exit $?"
grep -E "MB  blocks" /tmp/ow.log | head -8
[ -f /tmp/ow.json ] && echo "ONEWAY_JSON $(python -c "import json;print(json.dumps(json.load(open('/tmp/ow.json'))))")"
echo DONE
