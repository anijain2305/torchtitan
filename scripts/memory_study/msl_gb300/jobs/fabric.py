# Print each local GPU's NVLink fabric (cluster UUID, clique) so runs record whether hosts share an NVL72 domain.
import socket
import pynvml as n
n.nvmlInit()
for i in range(n.nvmlDeviceGetCount()):
    h = n.nvmlDeviceGetHandleByIndex(i)
    f = n.c_nvmlGpuFabricInfo_t()
    n.nvmlDeviceGetGpuFabricInfo(h, f)
    print(f"FABRIC {socket.gethostname()} gpu{i} cluster={bytes(f.clusterUuid).hex()} clique={f.cliqueId} state={f.state}")
