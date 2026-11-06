# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print each GPU's NVLink fabric (NVL72 domain) membership via NVML.

Needs only the system python3 and the driver's libnvidia-ml, so it runs in the
bare conda_mast_core image, which has no nvidia-smi and no PyTorch. GPUs with
the same ``cluster`` and ``clique`` are in the same NVLink domain.
``state=3`` is NVML_GPU_FABRIC_STATE_COMPLETED and ``status=0`` is success.
"""

import ctypes
import glob
import socket

paths = (
    ["libnvidia-ml.so.1"]
    + glob.glob("/usr/lib*/**/libnvidia-ml.so*", recursive=True)
    + glob.glob("/usr/local/**/libnvidia-ml.so*", recursive=True)
)
lib = None
for p in paths:
    try:
        lib = ctypes.CDLL(p)
        print("NVML", p)
        break
    except OSError:
        pass
if lib is None:
    raise SystemExit("NVML not found")


class Fabric(ctypes.Structure):
    # nvmlGpuFabricInfo_t
    _fields_ = [
        ("clusterUuid", ctypes.c_ubyte * 16),
        ("status", ctypes.c_uint),
        ("cliqueId", ctypes.c_uint),
        ("state", ctypes.c_ubyte),
    ]


assert lib.nvmlInit_v2() == 0
n = ctypes.c_uint()
lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
host = socket.gethostname().split(".")[1]
for i in range(n.value):
    h = ctypes.c_void_p()
    lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
    name = ctypes.create_string_buffer(96)
    lib.nvmlDeviceGetName(h, name, 96)
    f = Fabric()
    rc = lib.nvmlDeviceGetGpuFabricInfo(h, ctypes.byref(f))
    print(
        f"FABRIC host={host} gpu={i} {name.value.decode()} rc={rc} state={f.state} "
        f"status={f.status} cluster={bytes(f.clusterUuid).hex()} clique={f.cliqueId}"
    )
