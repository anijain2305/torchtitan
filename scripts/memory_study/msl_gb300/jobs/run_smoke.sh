cd "$(dirname "$0")"; source ./gpu_env.sh
hostname; python -c "import torch; print('torch', torch.__version__, torch.cuda.device_count())"
python fabric.py
MASTER=$(echo $MAST_HPC_TASK_GROUP_HOSTNAMES | cut -d, -f1); NN=$(echo $MAST_HPC_TASK_GROUP_HOSTNAMES | tr ',' '\n' | wc -l)
for mnnvl in default 0; do
  E=""; port=29501; [ $mnnvl = 0 ] && E="NCCL_MNNVL_ENABLE=0" && port=29502
  env $E NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH torchrun --nnodes $NN --nproc-per-node 4 --node-rank $TW_TASK_ID \
    --master-addr $MASTER --master-port $port nccl_probe.py > /tmp/probe_$mnnvl.log 2>&1
  echo "=== MNNVL=$mnnvl exit $? ($(wc -l < /tmp/probe_$mnnvl.log) lines)"
  grep -E "PROBE|MNNVL|NVLS|via NET|via P2P|NET/IB : Using|Using network|comm .* rank 0 nranks" /tmp/probe_$mnnvl.log | sed 's/^.*NCCL INFO //' | sort | uniq -c | sort -rn | head -14
  grep -iE "Traceback|Error" /tmp/probe_$mnnvl.log | head -5
done
