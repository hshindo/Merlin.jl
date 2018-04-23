const DEFINE = Dict{Symbol,Symbol}()

DEFINE[:nvmlInit] = :nvmlInit_v2
DEFINE[:nvmlDeviceGetPciInfo] = :nvmlDeviceGetPciInfo_v3
DEFINE[:nvmlDeviceGetCount] = :nvmlDeviceGetCount_v2
DEFINE[:nvmlDeviceGetHandleByIndex] = :nvmlDeviceGetHandleByIndex_v2
DEFINE[:nvmlDeviceGetHandleByPciBusId] = :nvmlDeviceGetHandleByPciBusId_v2
DEFINE[:nvmlDeviceGetNvLinkRemotePciInfo] = :nvmlDeviceGetNvLinkRemotePciInfo_v2
