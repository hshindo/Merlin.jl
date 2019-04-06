# cudnnDataType_t
const CUDNN_DATA_FLOAT = Cint(0)
const CUDNN_DATA_DOUBLE = Cint(1)
const CUDNN_DATA_HALF = Cint(2)
const CUDNN_DATA_INT8 = Cint(3)
const CUDNN_DATA_INT32 = Cint(4)
const CUDNN_DATA_INT8x4 = Cint(5)

const Cptr = Ptr{Cvoid}
datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF
datatype(::Type{Int8}) = CUDNN_DATA_INT8
datatype(::Type{Int32}) = CUDNN_DATA_INT32

# cudnnNanPropagation_t
const CUDNN_NOT_PROPAGATE_NAN = 0
const CUDNN_PROPAGATE_NAN = 1

# cudnnMathType_t
const CUDNN_DEFAULT_MATH = Cint(0)
const CUDNN_TENSOR_OP_MATH = Cint(1)
const CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = Cint(2)

# cudnnDeterminism_t
const CUDNN_NON_DETERMINISTIC = Cint(0)
const CUDNN_DETERMINISTIC = Cint(1)

# cudnnOpTensorOp_t
const CUDNN_OP_TENSOR_ADD = Cint(0)
const CUDNN_OP_TENSOR_MUL = Cint(1)
const CUDNN_OP_TENSOR_MIN = Cint(2)
const CUDNN_OP_TENSOR_MAX = Cint(3)
const CUDNN_OP_TENSOR_SQRT = Cint(4)
const CUDNN_OP_TENSOR_NOT = Cint(5)

# cudnnIndicesType_t
const CUDNN_32BIT_INDICES = Cint(0)
const CUDNN_64BIT_INDICES = Cint(1)
const CUDNN_16BIT_INDICES = Cint(2)
const CUDNN_8BIT_INDICES = Cint(3)

# cudnnLRNMode_t
const CUDNN_LRN_CROSS_CHANNEL_DIM1 = Cint(0)

# cudnnDivNormMode_t
const CUDNN_DIVNORM_PRECOMPUTED_MEANS = Cint(0)

# cudnnSamplerType_t
const CUDNN_SAMPLER_BILINEAR = Cint(0)

# cudnnCTCLossAlgo_t
const CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = Cint(0)
const CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = Cint(1)
