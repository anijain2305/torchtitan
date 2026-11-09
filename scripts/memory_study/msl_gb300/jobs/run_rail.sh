# One GPU per host in each group (as an expert shard's FSDP replicas across NVLink
# domains), all groups at once, across InfiniBand; PXN off (cluster default) and on.
cd "$(dirname "$0")"; source ./gpu_env.sh
TR="torchrun --nnodes $NN --nproc-per-node 4 --node-rank $NODE_RANK --master-addr $MASTER"
export NCCL_MNNVL_ENABLE=0 NCCL_DEBUG=WARN
BW_TAG=pxn_off NCCL_PXN_DISABLE=1 timeout 900 $TR --master-port 29911 rail_bw.py 2>&1 | grep "^RAIL "
BW_TAG=pxn_on NCCL_PXN_DISABLE=0 timeout 900 $TR --master-port 29912 rail_bw.py 2>&1 | grep "^RAIL "
echo DONE
