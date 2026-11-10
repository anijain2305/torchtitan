# Multi-host NVLink (MNNVL) inside one NVL72 domain: fabric IDs, NCCL transport,
# collective bandwidth, then collective overlap over every GPU of the job.
cd "$(dirname "$0")"; source ./gpu_env.sh
TR="torchrun --nnodes $NN --nproc-per-node 4 --node-rank $NODE_RANK --master-addr $MASTER"
python nvlink_fabric_info.py | grep FABRIC
echo "IMEX $(ls /dev/nvidia-caps-imex-channels 2>&1 | tr '\n' ' ')"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET timeout 300 $TR --master-port 29701 nccl_probe.py > /tmp/probe.log 2>&1
rc=$?; echo "=== probe exit $rc"
grep -E "PROBE|MNNVL|via NET|via P2P|NVLS|Using network|NET/IB : Using" /tmp/probe.log | sed 's/^.*NCCL INFO //' \
  | sed 's/\[[0-9]*\]//g' | sort | uniq -c | sort -rn | head -20
[ $rc = 0 ] || { grep -iE "warn|error" /tmp/probe.log | head -20; tail -20 /tmp/probe.log; exit 1; }
BW_TAG=nvl_${NN}hosts NCCL_DEBUG=WARN timeout 1200 $TR --master-port 29702 coll_bw.py 2>&1 | grep "^BW "
BW_TAG=nvl_nvls_${NN}hosts NCCL_NVLS_ENABLE=1 NCCL_DEBUG=WARN timeout 1200 $TR --master-port 29703 coll_bw.py 2>&1 | grep "^BW "
NCCL_DEBUG=WARN timeout 1800 $TR --master-port 29704 kimi_k3_overlap_bench.py --sizes-mb 256 1024 \
  --out /tmp/overlap_nvl.json > /tmp/overlap_nvl.log 2>&1
echo "=== overlap exit $?"
grep -E "MB  blocks|Wrote" /tmp/overlap_nvl.log
grep -E "Traceback|Error" /tmp/overlap_nvl.log | grep -v "error_handler_fb\|ERROR_HANDLING\|RETHROW\|ScubaLogger" | head -5
[ -f /tmp/overlap_nvl.json ] && echo "OVERLAP_JSON $(python -c "import json;print(json.dumps(json.load(open('/tmp/overlap_nvl.json'))))")"
echo DONE
