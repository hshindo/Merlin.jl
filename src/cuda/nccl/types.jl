const NCCL_UNIQUE_ID_BYTES = 128

# ncclDataType_t
const ncclInt8 = 0
const ncclChar = 0
const ncclUint8 = 1
const ncclInt32 = 2
const ncclInt = 2
const ncclUint32 = 3
const ncclInt64 = 4
const ncclUint64 = 5
const ncclFloat16 = 6
const ncclHalf = 6,
const ncclFloat32 = 7
const ncclFloat = 7
const ncclFloat64 = 8
const ncclDouble = 8
const ncclNumTypes = 9

# ncclRedOp_t
const ncclSum = 0
const ncclProd = 1
const ncclMax = 2
const ncclMin = 3
const ncclNumOps = 4
