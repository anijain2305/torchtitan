# Islands of 8 GPUs: 4 hosts in one NVL72 domain, split into two NVLink cliques
# (hosts 0-1, hosts 2-3) with NCCL_MNNVL_CLIQUE_ID, so NCCL uses multi-host
# NVLink inside an island and InfiniBand between them.
cd "$(dirname "$0")"; source ./gpu_env.sh
TR="torchrun --nnodes $NN --nproc-per-node 4 --node-rank $NODE_RANK --master-addr $MASTER"
export NCCL_MNNVL_CLIQUE_ID=$((1 + NODE_RANK / 2)) NCCL_MNNVL_CROSS_CLIQUE=0
python nvlink_fabric_info.py | grep FABRIC | head -1
echo "CLIQUE node $NODE_RANK -> $NCCL_MNNVL_CLIQUE_ID"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL timeout 600 $TR --master-port 29921 algo_probe.py > /tmp/algo.log 2>&1
echo "=== probe exit $?"
sed 's/^.*NCCL INFO //' /tmp/algo.log | grep -E "(AllGather|ReduceScatter): [0-9]+ Bytes -> Algo|coll channels|nNodes|MNNVL [01] cliqueId|Channel 00/0 :|Ring 00 :|Pattern" \
  | sed 's/comm 0x[0-9a-f]*//' | sort | uniq -c | sort -rn | head -40
export NCCL_DEBUG=WARN
BW_TAG=island8 BW_OPS=allgather,reducescatter,sendrecv timeout 900 $TR --master-port 29922 coll_bw.py 2>&1 | grep "^BW "
BW_TAG=island8_rail timeout 900 $TR --master-port 29923 rail_bw.py 2>&1 | grep "^RAIL "
timeout 1500 $TR --master-port 29924 kimi_k3_overlap_bench.py --sizes-mb 1024 --collectives all_gather reduce_scatter \
  --out /tmp/ov.json > /tmp/ov.log 2>&1
echo "=== overlap exit $?"
grep -E "MB  blocks" /tmp/ov.log
[ -f /tmp/ov.json ] && echo "OVERLAP_JSON $(python -c "import json;print(json.dumps(json.load(open('/tmp/ov.json'))))")"
echo DONE
