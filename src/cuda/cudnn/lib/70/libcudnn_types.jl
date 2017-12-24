# Automatically generated using Clang.jl wrap_c, version 0.0.0

using Compat

const CUDNN_MAJOR = 7
const CUDNN_MINOR = 0
const CUDNN_PATCHLEVEL = 5

# Skipping MacroDefinition: CUDNN_VERSION ( CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL ) #

const CUDNN_DIM_MAX = 8
const CUDNN_LRN_MIN_N = 1
const CUDNN_LRN_MAX_N = 16
const CUDNN_LRN_MIN_K = 1.0e-5
const CUDNN_LRN_MIN_BETA = 0.01
const CUDNN_BN_MIN_EPSILON = 1.0e-5
const cudnnContext = Void
const cudnnHandle_t = Ptr{Void}

# begin enum ANONYMOUS_1
const ANONYMOUS_1 = UInt32
const CUDNN_STATUS_SUCCESS = (UInt32)(0)
const CUDNN_STATUS_NOT_INITIALIZED = (UInt32)(1)
const CUDNN_STATUS_ALLOC_FAILED = (UInt32)(2)
const CUDNN_STATUS_BAD_PARAM = (UInt32)(3)
const CUDNN_STATUS_INTERNAL_ERROR = (UInt32)(4)
const CUDNN_STATUS_INVALID_VALUE = (UInt32)(5)
const CUDNN_STATUS_ARCH_MISMATCH = (UInt32)(6)
const CUDNN_STATUS_MAPPING_ERROR = (UInt32)(7)
const CUDNN_STATUS_EXECUTION_FAILED = (UInt32)(8)
const CUDNN_STATUS_NOT_SUPPORTED = (UInt32)(9)
const CUDNN_STATUS_LICENSE_ERROR = (UInt32)(10)
const CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = (UInt32)(11)
const CUDNN_STATUS_RUNTIME_IN_PROGRESS = (UInt32)(12)
const CUDNN_STATUS_RUNTIME_FP_OVERFLOW = (UInt32)(13)
# end enum ANONYMOUS_1

const cudnnStatus_t = Void
const cudnnRuntimeTag_t = Void

# begin enum ANONYMOUS_2
const ANONYMOUS_2 = UInt32
const CUDNN_ERRQUERY_RAWCODE = (UInt32)(0)
const CUDNN_ERRQUERY_NONBLOCKING = (UInt32)(1)
const CUDNN_ERRQUERY_BLOCKING = (UInt32)(2)
# end enum ANONYMOUS_2

const cudnnErrQueryMode_t = Void
const cudnnTensorStruct = Void
const cudnnTensorDescriptor_t = Ptr{Void}
const cudnnConvolutionStruct = Void
const cudnnConvolutionDescriptor_t = Ptr{Void}
const cudnnPoolingStruct = Void
const cudnnPoolingDescriptor_t = Ptr{Void}
const cudnnFilterStruct = Void
const cudnnFilterDescriptor_t = Ptr{Void}
const cudnnLRNStruct = Void
const cudnnLRNDescriptor_t = Ptr{Void}
const cudnnActivationStruct = Void
const cudnnActivationDescriptor_t = Ptr{Void}
const cudnnSpatialTransformerStruct = Void
const cudnnSpatialTransformerDescriptor_t = Ptr{Void}
const cudnnOpTensorStruct = Void
const cudnnOpTensorDescriptor_t = Ptr{Void}
const cudnnReduceTensorStruct = Void
const cudnnReduceTensorDescriptor_t = Ptr{Void}
const cudnnCTCLossStruct = Void
const cudnnCTCLossDescriptor_t = Ptr{Void}

# begin enum ANONYMOUS_3
const ANONYMOUS_3 = UInt32
const CUDNN_DATA_FLOAT = (UInt32)(0)
const CUDNN_DATA_DOUBLE = (UInt32)(1)
const CUDNN_DATA_HALF = (UInt32)(2)
const CUDNN_DATA_INT8 = (UInt32)(3)
const CUDNN_DATA_INT32 = (UInt32)(4)
const CUDNN_DATA_INT8x4 = (UInt32)(5)
# end enum ANONYMOUS_3

const cudnnDataType_t = Void

# begin enum ANONYMOUS_4
const ANONYMOUS_4 = UInt32
const CUDNN_DEFAULT_MATH = (UInt32)(0)
const CUDNN_TENSOR_OP_MATH = (UInt32)(1)
# end enum ANONYMOUS_4

const cudnnMathType_t = Void

# begin enum ANONYMOUS_5
const ANONYMOUS_5 = UInt32
const CUDNN_NOT_PROPAGATE_NAN = (UInt32)(0)
const CUDNN_PROPAGATE_NAN = (UInt32)(1)
# end enum ANONYMOUS_5

const cudnnNanPropagation_t = Void

# begin enum ANONYMOUS_6
const ANONYMOUS_6 = UInt32
const CUDNN_NON_DETERMINISTIC = (UInt32)(0)
const CUDNN_DETERMINISTIC = (UInt32)(1)
# end enum ANONYMOUS_6

const cudnnDeterminism_t = Void

# begin enum ANONYMOUS_7
const ANONYMOUS_7 = UInt32
const CUDNN_TENSOR_NCHW = (UInt32)(0)
const CUDNN_TENSOR_NHWC = (UInt32)(1)
const CUDNN_TENSOR_NCHW_VECT_C = (UInt32)(2)
# end enum ANONYMOUS_7

const cudnnTensorFormat_t = Void

# begin enum ANONYMOUS_8
const ANONYMOUS_8 = UInt32
const CUDNN_OP_TENSOR_ADD = (UInt32)(0)
const CUDNN_OP_TENSOR_MUL = (UInt32)(1)
const CUDNN_OP_TENSOR_MIN = (UInt32)(2)
const CUDNN_OP_TENSOR_MAX = (UInt32)(3)
const CUDNN_OP_TENSOR_SQRT = (UInt32)(4)
const CUDNN_OP_TENSOR_NOT = (UInt32)(5)
# end enum ANONYMOUS_8

const cudnnOpTensorOp_t = Void

# begin enum ANONYMOUS_9
const ANONYMOUS_9 = UInt32
const CUDNN_REDUCE_TENSOR_ADD = (UInt32)(0)
const CUDNN_REDUCE_TENSOR_MUL = (UInt32)(1)
const CUDNN_REDUCE_TENSOR_MIN = (UInt32)(2)
const CUDNN_REDUCE_TENSOR_MAX = (UInt32)(3)
const CUDNN_REDUCE_TENSOR_AMAX = (UInt32)(4)
const CUDNN_REDUCE_TENSOR_AVG = (UInt32)(5)
const CUDNN_REDUCE_TENSOR_NORM1 = (UInt32)(6)
const CUDNN_REDUCE_TENSOR_NORM2 = (UInt32)(7)
const CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = (UInt32)(8)
# end enum ANONYMOUS_9

const cudnnReduceTensorOp_t = Void

# begin enum ANONYMOUS_10
const ANONYMOUS_10 = UInt32
const CUDNN_REDUCE_TENSOR_NO_INDICES = (UInt32)(0)
const CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = (UInt32)(1)
# end enum ANONYMOUS_10

const cudnnReduceTensorIndices_t = Void

# begin enum ANONYMOUS_11
const ANONYMOUS_11 = UInt32
const CUDNN_32BIT_INDICES = (UInt32)(0)
const CUDNN_64BIT_INDICES = (UInt32)(1)
const CUDNN_16BIT_INDICES = (UInt32)(2)
const CUDNN_8BIT_INDICES = (UInt32)(3)
# end enum ANONYMOUS_11

const cudnnIndicesType_t = Void

# begin enum ANONYMOUS_12
const ANONYMOUS_12 = UInt32
const CUDNN_CONVOLUTION = (UInt32)(0)
const CUDNN_CROSS_CORRELATION = (UInt32)(1)
# end enum ANONYMOUS_12

const cudnnConvolutionMode_t = Void

# begin enum ANONYMOUS_13
const ANONYMOUS_13 = UInt32
const CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum ANONYMOUS_13

const cudnnConvolutionFwdPreference_t = Void

# begin enum ANONYMOUS_14
const ANONYMOUS_14 = UInt32
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = (UInt32)(0)
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = (UInt32)(1)
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = (UInt32)(2)
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = (UInt32)(3)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT = (UInt32)(4)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = (UInt32)(5)
const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = (UInt32)(6)
const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = (UInt32)(7)
const CUDNN_CONVOLUTION_FWD_ALGO_COUNT = (UInt32)(8)
# end enum ANONYMOUS_14

const cudnnConvolutionFwdAlgo_t = Void
const cudnnConvolutionFwdAlgoPerf_t = Void

# begin enum ANONYMOUS_15
const ANONYMOUS_15 = UInt32
const CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum ANONYMOUS_15

const cudnnConvolutionBwdFilterPreference_t = Void

# begin enum ANONYMOUS_16
const ANONYMOUS_16 = UInt32
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = (UInt32)(2)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = (UInt32)(3)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = (UInt32)(4)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = (UInt32)(5)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = (UInt32)(6)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = (UInt32)(7)
# end enum ANONYMOUS_16

const cudnnConvolutionBwdFilterAlgo_t = Void
const cudnnConvolutionBwdFilterAlgoPerf_t = Void

# begin enum ANONYMOUS_17
const ANONYMOUS_17 = UInt32
const CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum ANONYMOUS_17

const cudnnConvolutionBwdDataPreference_t = Void

# begin enum ANONYMOUS_18
const ANONYMOUS_18 = UInt32
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = (UInt32)(2)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = (UInt32)(3)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = (UInt32)(4)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = (UInt32)(5)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = (UInt32)(6)
# end enum ANONYMOUS_18

const cudnnConvolutionBwdDataAlgo_t = Void
const cudnnConvolutionBwdDataAlgoPerf_t = Void

# begin enum ANONYMOUS_19
const ANONYMOUS_19 = UInt32
const CUDNN_SOFTMAX_FAST = (UInt32)(0)
const CUDNN_SOFTMAX_ACCURATE = (UInt32)(1)
const CUDNN_SOFTMAX_LOG = (UInt32)(2)
# end enum ANONYMOUS_19

const cudnnSoftmaxAlgorithm_t = Void

# begin enum ANONYMOUS_20
const ANONYMOUS_20 = UInt32
const CUDNN_SOFTMAX_MODE_INSTANCE = (UInt32)(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = (UInt32)(1)
# end enum ANONYMOUS_20

const cudnnSoftmaxMode_t = Void

# begin enum ANONYMOUS_21
const ANONYMOUS_21 = UInt32
const CUDNN_POOLING_MAX = (UInt32)(0)
const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = (UInt32)(1)
const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = (UInt32)(2)
const CUDNN_POOLING_MAX_DETERMINISTIC = (UInt32)(3)
# end enum ANONYMOUS_21

const cudnnPoolingMode_t = Void

# begin enum ANONYMOUS_22
const ANONYMOUS_22 = UInt32
const CUDNN_ACTIVATION_SIGMOID = (UInt32)(0)
const CUDNN_ACTIVATION_RELU = (UInt32)(1)
const CUDNN_ACTIVATION_TANH = (UInt32)(2)
const CUDNN_ACTIVATION_CLIPPED_RELU = (UInt32)(3)
const CUDNN_ACTIVATION_ELU = (UInt32)(4)
# end enum ANONYMOUS_22

const cudnnActivationMode_t = Void

# begin enum ANONYMOUS_23
const ANONYMOUS_23 = UInt32
const CUDNN_LRN_CROSS_CHANNEL_DIM1 = (UInt32)(0)
# end enum ANONYMOUS_23

const cudnnLRNMode_t = Void

# begin enum ANONYMOUS_24
const ANONYMOUS_24 = UInt32
const CUDNN_DIVNORM_PRECOMPUTED_MEANS = (UInt32)(0)
# end enum ANONYMOUS_24

const cudnnDivNormMode_t = Void

# begin enum ANONYMOUS_25
const ANONYMOUS_25 = UInt32
const CUDNN_BATCHNORM_PER_ACTIVATION = (UInt32)(0)
const CUDNN_BATCHNORM_SPATIAL = (UInt32)(1)
const CUDNN_BATCHNORM_SPATIAL_PERSISTENT = (UInt32)(2)
# end enum ANONYMOUS_25

const cudnnBatchNormMode_t = Void

# begin enum ANONYMOUS_26
const ANONYMOUS_26 = UInt32
const CUDNN_SAMPLER_BILINEAR = (UInt32)(0)
# end enum ANONYMOUS_26

const cudnnSamplerType_t = Void
const cudnnDropoutStruct = Void
const cudnnDropoutDescriptor_t = Ptr{Void}

# begin enum ANONYMOUS_27
const ANONYMOUS_27 = UInt32
const CUDNN_RNN_RELU = (UInt32)(0)
const CUDNN_RNN_TANH = (UInt32)(1)
const CUDNN_LSTM = (UInt32)(2)
const CUDNN_GRU = (UInt32)(3)
# end enum ANONYMOUS_27

const cudnnRNNMode_t = Void

# begin enum ANONYMOUS_28
const ANONYMOUS_28 = UInt32
const CUDNN_UNIDIRECTIONAL = (UInt32)(0)
const CUDNN_BIDIRECTIONAL = (UInt32)(1)
# end enum ANONYMOUS_28

const cudnnDirectionMode_t = Void

# begin enum ANONYMOUS_29
const ANONYMOUS_29 = UInt32
const CUDNN_LINEAR_INPUT = (UInt32)(0)
const CUDNN_SKIP_INPUT = (UInt32)(1)
# end enum ANONYMOUS_29

const cudnnRNNInputMode_t = Void

# begin enum ANONYMOUS_30
const ANONYMOUS_30 = UInt32
const CUDNN_RNN_ALGO_STANDARD = (UInt32)(0)
const CUDNN_RNN_ALGO_PERSIST_STATIC = (UInt32)(1)
const CUDNN_RNN_ALGO_PERSIST_DYNAMIC = (UInt32)(2)
# end enum ANONYMOUS_30

const cudnnRNNAlgo_t = Void
const cudnnRNNStruct = Void
const cudnnRNNDescriptor_t = Ptr{Void}
const cudnnPersistentRNNPlan = Void
const cudnnPersistentRNNPlan_t = Ptr{Void}

# begin enum ANONYMOUS_31
const ANONYMOUS_31 = UInt32
const CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = (UInt32)(0)
const CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = (UInt32)(1)
# end enum ANONYMOUS_31

const cudnnCTCLossAlgo_t = Void
