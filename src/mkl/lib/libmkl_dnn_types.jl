# Automatically generated using Clang.jl wrap_c, version 0.0.0

#using Compat

const DNN_MAX_DIMENSION = 32
const DNN_QUERY_MAX_LENGTH = 128

typealias _uniPrimitive_s Void
typealias dnnPrimitive_t Ptr{_uniPrimitive_s}
typealias _dnnLayout_s Void
typealias dnnLayout_t Ptr{_dnnLayout_s}
typealias dnnPrimitiveAttributes_t Ptr{Cvoid}

# begin enum ANONYMOUS_1
typealias ANONYMOUS_1 Cint
const E_SUCCESS = (Int32)(0)
const E_INCORRECT_INPUT_PARAMETER = (Int32)(-1)
const E_UNEXPECTED_NULL_POINTER = (Int32)(-2)
const E_MEMORY_ERROR = (Int32)(-3)
const E_UNSUPPORTED_DIMENSION = (Int32)(-4)
const E_UNIMPLEMENTED = (Int32)(-127)
# end enum ANONYMOUS_1

# begin enum dnnError_t
typealias dnnError_t Cint
const E_SUCCESS = (Int32)(0)
const E_INCORRECT_INPUT_PARAMETER = (Int32)(-1)
const E_UNEXPECTED_NULL_POINTER = (Int32)(-2)
const E_MEMORY_ERROR = (Int32)(-3)
const E_UNSUPPORTED_DIMENSION = (Int32)(-4)
const E_UNIMPLEMENTED = (Int32)(-127)
# end enum dnnError_t

# begin enum ANONYMOUS_2
typealias ANONYMOUS_2 UInt32
const dnnAlgorithmConvolutionGemm = (UInt32)(0)
const dnnAlgorithmConvolutionDirect = (UInt32)(1)
const dnnAlgorithmConvolutionFFT = (UInt32)(2)
const dnnAlgorithmPoolingMax = (UInt32)(3)
const dnnAlgorithmPoolingMin = (UInt32)(4)
const dnnAlgorithmPoolingAvg = (UInt32)(5)
# end enum ANONYMOUS_2

# begin enum dnnAlgorithm_t
typealias dnnAlgorithm_t UInt32
const dnnAlgorithmConvolutionGemm = (UInt32)(0)
const dnnAlgorithmConvolutionDirect = (UInt32)(1)
const dnnAlgorithmConvolutionFFT = (UInt32)(2)
const dnnAlgorithmPoolingMax = (UInt32)(3)
const dnnAlgorithmPoolingMin = (UInt32)(4)
const dnnAlgorithmPoolingAvg = (UInt32)(5)
# end enum dnnAlgorithm_t

# begin enum ANONYMOUS_3
typealias ANONYMOUS_3 UInt32
const dnnResourceSrc = (UInt32)(0)
const dnnResourceFrom = (UInt32)(0)
const dnnResourceDst = (UInt32)(1)
const dnnResourceTo = (UInt32)(1)
const dnnResourceFilter = (UInt32)(2)
const dnnResourceScaleShift = (UInt32)(2)
const dnnResourceBias = (UInt32)(3)
const dnnResourceDiffSrc = (UInt32)(4)
const dnnResourceDiffFilter = (UInt32)(5)
const dnnResourceDiffScaleShift = (UInt32)(5)
const dnnResourceDiffBias = (UInt32)(6)
const dnnResourceDiffDst = (UInt32)(7)
const dnnResourceWorkspace = (UInt32)(8)
const dnnResourceMultipleSrc = (UInt32)(16)
const dnnResourceMultipleDst = (UInt32)(24)
const dnnResourceNumber = (UInt32)(32)
# end enum ANONYMOUS_3

# begin enum dnnResourceType_t
typealias dnnResourceType_t UInt32
const dnnResourceSrc = (UInt32)(0)
const dnnResourceFrom = (UInt32)(0)
const dnnResourceDst = (UInt32)(1)
const dnnResourceTo = (UInt32)(1)
const dnnResourceFilter = (UInt32)(2)
const dnnResourceScaleShift = (UInt32)(2)
const dnnResourceBias = (UInt32)(3)
const dnnResourceDiffSrc = (UInt32)(4)
const dnnResourceDiffFilter = (UInt32)(5)
const dnnResourceDiffScaleShift = (UInt32)(5)
const dnnResourceDiffBias = (UInt32)(6)
const dnnResourceDiffDst = (UInt32)(7)
const dnnResourceWorkspace = (UInt32)(8)
const dnnResourceMultipleSrc = (UInt32)(16)
const dnnResourceMultipleDst = (UInt32)(24)
const dnnResourceNumber = (UInt32)(32)
# end enum dnnResourceType_t

# begin enum ANONYMOUS_4
typealias ANONYMOUS_4 UInt32
const dnnBorderZeros = (UInt32)(0)
const dnnBorderExtrapolation = (UInt32)(3)
# end enum ANONYMOUS_4

# begin enum dnnBorder_t
typealias dnnBorder_t UInt32
const dnnBorderZeros = (UInt32)(0)
const dnnBorderExtrapolation = (UInt32)(3)
# end enum dnnBorder_t
