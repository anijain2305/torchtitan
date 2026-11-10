# The GB300 host's NVIDIA driver libraries live in the fbcode platform dir, off the
# loader path. Link only those (not the platform's libc etc.) into a private dir.
D=/usr/local/fbcode/platform010-aarch64/lib
mkdir -p /tmp/nvdrv && for f in $D/libcuda.so* $D/libnvidia*.so*; do [ -e "$f" ] && ln -sf "$f" /tmp/nvdrv/; done
export LD_LIBRARY_PATH=/tmp/nvdrv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# RDMA: the system provider /usr/lib64/libibverbs/libmlx5-rdmav57.so is a dangling
# symlink (to a libmlx5 1.25.57 that only the fbcode platform dir has), so verbs
# finds 0 devices and NCCL's IB init fails. Use the platform dir's matching pair
# (libibverbs 1.14.57 + libmlx5 1.25.57), linked into a private dir first on the path.
mkdir -p /tmp/ibv
ln -sf $D/libibverbs.so.1.14.57.0 /tmp/ibv/libibverbs.so.1
ln -sf $D/libibverbs.so.1.14.57.0 /tmp/ibv/libibverbs.so
ln -sf $D/libmlx5.so.1.25.57.0 /tmp/ibv/libmlx5.so.1
ln -sf $D/libmlx5.so.1.25.57.0 /tmp/ibv/libmlx5-rdmav57.so
export LD_LIBRARY_PATH=/tmp/ibv:$LD_LIBRARY_PATH
# Rendezvous: node rank = this host's position in MAST_HPC_TASK_GROUP_HOSTNAMES
# (TW_TASK_ID need not follow that order), master = the first host.
MASTER=$(echo $MAST_HPC_TASK_GROUP_HOSTNAMES | cut -d, -f1)
NN=$(echo $MAST_HPC_TASK_GROUP_HOSTNAMES | tr ',' '\n' | wc -l)
ME=$(hostname -f 2>/dev/null || hostname)
NODE_RANK=$(echo $MAST_HPC_TASK_GROUP_HOSTNAMES | tr ',' '\n' | grep -nx "$ME" | cut -d: -f1)
NODE_RANK=$(( ${NODE_RANK:-$((TW_TASK_ID + 1))} - 1 ))
echo "RDZV host=$ME node_rank=$NODE_RANK task=$TW_TASK_ID master=$MASTER nn=$NN"
