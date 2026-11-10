# Collective bus bandwidth across every GPU of the job, over InfiniBand between hosts.
cd "$(dirname "$0")"; source ./gpu_env.sh

BW_TAG=ib_${NN}hosts NCCL_MNNVL_ENABLE=0 NCCL_DEBUG=WARN timeout 1500 torchrun --nnodes $NN --nproc-per-node 4 \
  --node-rank $NODE_RANK --master-addr $MASTER --master-port 29612 coll_bw.py 2>&1 | grep "^BW "
echo DONE
