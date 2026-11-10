#!/bin/bash
# Usage: [DOMAIN=1] submit.sh <num_hosts> <script in jobs/> [--dryrun]
# DOMAIN=1 places all hosts in one NVL72 NVLink domain.
# Packs the msl_bench conda env (with jobs/ and the overlap benchmark inside it)
# into an ephemeral metaconda_demo fbpkg and submits to the MSL GB300 tenant.
# See NOTES.md.
W=$(cd "$(dirname "$0")" && pwd)/jobs  # job scripts; the benchmark is copied in below
eval "$(conda shell.bash hook)"
conda activate msl_bench && echo "ACTIVE $CONDA_PREFIX" || exit 3
[ "$CONDA_PREFIX" = /home/anijain/.conda/envs/msl_bench ] || { echo "wrong env $CONDA_PREFIX"; exit 4; }
# The benchmark scripts travel inside the conda env (msl_ws/), so one ephemeral
# metaconda_demo fbpkg carries everything; sync them in before packing.
mkdir -p $CONDA_PREFIX/msl_ws && cp $W/*.py $W/*.sh $W/../../kimi_k3_overlap_bench.py $W/../nvlink_fabric_info.py $CONDA_PREFIX/msl_ws/
SA="hpcIdentity=pytorch_distributed,hpcClusterUuid=MastGenAICluster,rmAttribution=msl_infra_pytorch_dct,hpcJobOncall=pytorch_distributed,localityConstraints=region;ofn,jobPriority=LOW,conda_fbpkg_name=metaconda_demo,activate_conda=True,workspace_dir=/packages/metaconda_demo/conda/msl_ws"
cd $W
torchx run $3 --scheduler msl_conda --scheduler_args "$SA" $W/raw_sh_env.py:raw_sh_env --num_replicas $1 ${DOMAIN:+--one_scaleup_domain True} --script "bash $2"
