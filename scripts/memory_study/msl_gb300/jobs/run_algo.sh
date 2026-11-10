# NCCL's algorithm choice and ring layout for all-gather / reduce-scatter across
# InfiniBand (multi-host NVLink off), from NCCL's own log.
cd "$(dirname "$0")"; source ./gpu_env.sh
NCCL_MNNVL_ENABLE=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL timeout 600 \
  torchrun --nnodes $NN --nproc-per-node 4 --node-rank $NODE_RANK --master-addr $MASTER --master-port 29901 \
  algo_probe.py > /tmp/algo.log 2>&1
echo "=== exit $?"
grep -E "ALGO_PROBE" /tmp/algo.log
sed 's/^.*NCCL INFO //' /tmp/algo.log | grep -E "^(AllGather|ReduceScatter)|algorithm|Algo|nChannels|channels|Channel 00/|Ring 00|Trees|PAT|NVLS|CollNet|Connected all rings|Pattern|comm .* nRanks" \
  | sed 's/opCount [0-9a-f]*//; s/comm 0x[0-9a-f]*//; s/sendbuff 0x[0-9a-f]* recvbuff 0x[0-9a-f]*//; s/stream 0x[0-9a-f]*//' \
  | sort | uniq -c | sort -rn | head -40
