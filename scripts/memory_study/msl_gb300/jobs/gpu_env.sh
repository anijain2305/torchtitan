# libibverbs loads its providers (libmlx5-rdmav57.so) by bare name: put their
# directory on the loader path, or verbs finds 0 devices and NCCL's IB init fails.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/libibverbs
