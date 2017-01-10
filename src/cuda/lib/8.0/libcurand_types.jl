# Automatically generated using Clang.jl wrap_c, version 0.0.0

#using Compat

# begin enum curandStatus
typealias curandStatus UInt32
const CURAND_STATUS_SUCCESS = (UInt32)(0)
const CURAND_STATUS_VERSION_MISMATCH = (UInt32)(100)
const CURAND_STATUS_NOT_INITIALIZED = (UInt32)(101)
const CURAND_STATUS_ALLOCATION_FAILED = (UInt32)(102)
const CURAND_STATUS_TYPE_ERROR = (UInt32)(103)
const CURAND_STATUS_OUT_OF_RANGE = (UInt32)(104)
const CURAND_STATUS_LENGTH_NOT_MULTIPLE = (UInt32)(105)
const CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = (UInt32)(106)
const CURAND_STATUS_LAUNCH_FAILURE = (UInt32)(201)
const CURAND_STATUS_PREEXISTING_FAILURE = (UInt32)(202)
const CURAND_STATUS_INITIALIZATION_FAILED = (UInt32)(203)
const CURAND_STATUS_ARCH_MISMATCH = (UInt32)(204)
const CURAND_STATUS_INTERNAL_ERROR = (UInt32)(999)
# end enum curandStatus

typealias curandStatus_t curandStatus

# begin enum curandRngType
typealias curandRngType UInt32
const CURAND_RNG_TEST = (UInt32)(0)
const CURAND_RNG_PSEUDO_DEFAULT = (UInt32)(100)
const CURAND_RNG_PSEUDO_XORWOW = (UInt32)(101)
const CURAND_RNG_PSEUDO_MRG32K3A = (UInt32)(121)
const CURAND_RNG_PSEUDO_MTGP32 = (UInt32)(141)
const CURAND_RNG_PSEUDO_MT19937 = (UInt32)(142)
const CURAND_RNG_PSEUDO_PHILOX4_32_10 = (UInt32)(161)
const CURAND_RNG_QUASI_DEFAULT = (UInt32)(200)
const CURAND_RNG_QUASI_SOBOL32 = (UInt32)(201)
const CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = (UInt32)(202)
const CURAND_RNG_QUASI_SOBOL64 = (UInt32)(203)
const CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = (UInt32)(204)
# end enum curandRngType

typealias curandRngType_t curandRngType

# begin enum curandOrdering
typealias curandOrdering UInt32
const CURAND_ORDERING_PSEUDO_BEST = (UInt32)(100)
const CURAND_ORDERING_PSEUDO_DEFAULT = (UInt32)(101)
const CURAND_ORDERING_PSEUDO_SEEDED = (UInt32)(102)
const CURAND_ORDERING_QUASI_DEFAULT = (UInt32)(201)
# end enum curandOrdering

typealias curandOrdering_t curandOrdering

# begin enum curandDirectionVectorSet
typealias curandDirectionVectorSet UInt32
const CURAND_DIRECTION_VECTORS_32_JOEKUO6 = (UInt32)(101)
const CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = (UInt32)(102)
const CURAND_DIRECTION_VECTORS_64_JOEKUO6 = (UInt32)(103)
const CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = (UInt32)(104)
# end enum curandDirectionVectorSet

typealias curandDirectionVectorSet_t curandDirectionVectorSet

immutable Array_32_UInt32
    d1::UInt32
    d2::UInt32
    d3::UInt32
    d4::UInt32
    d5::UInt32
    d6::UInt32
    d7::UInt32
    d8::UInt32
    d9::UInt32
    d10::UInt32
    d11::UInt32
    d12::UInt32
    d13::UInt32
    d14::UInt32
    d15::UInt32
    d16::UInt32
    d17::UInt32
    d18::UInt32
    d19::UInt32
    d20::UInt32
    d21::UInt32
    d22::UInt32
    d23::UInt32
    d24::UInt32
    d25::UInt32
    d26::UInt32
    d27::UInt32
    d28::UInt32
    d29::UInt32
    d30::UInt32
    d31::UInt32
    d32::UInt32
end

zero(::Type{Array_32_UInt32}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_32_UInt32(fill(zero(UInt32),32)...)
    end

typealias curandDirectionVectors32_t Array_32_UInt32

immutable Array_64_Culonglong
    d1::Culonglong
    d2::Culonglong
    d3::Culonglong
    d4::Culonglong
    d5::Culonglong
    d6::Culonglong
    d7::Culonglong
    d8::Culonglong
    d9::Culonglong
    d10::Culonglong
    d11::Culonglong
    d12::Culonglong
    d13::Culonglong
    d14::Culonglong
    d15::Culonglong
    d16::Culonglong
    d17::Culonglong
    d18::Culonglong
    d19::Culonglong
    d20::Culonglong
    d21::Culonglong
    d22::Culonglong
    d23::Culonglong
    d24::Culonglong
    d25::Culonglong
    d26::Culonglong
    d27::Culonglong
    d28::Culonglong
    d29::Culonglong
    d30::Culonglong
    d31::Culonglong
    d32::Culonglong
    d33::Culonglong
    d34::Culonglong
    d35::Culonglong
    d36::Culonglong
    d37::Culonglong
    d38::Culonglong
    d39::Culonglong
    d40::Culonglong
    d41::Culonglong
    d42::Culonglong
    d43::Culonglong
    d44::Culonglong
    d45::Culonglong
    d46::Culonglong
    d47::Culonglong
    d48::Culonglong
    d49::Culonglong
    d50::Culonglong
    d51::Culonglong
    d52::Culonglong
    d53::Culonglong
    d54::Culonglong
    d55::Culonglong
    d56::Culonglong
    d57::Culonglong
    d58::Culonglong
    d59::Culonglong
    d60::Culonglong
    d61::Culonglong
    d62::Culonglong
    d63::Culonglong
    d64::Culonglong
end

zero(::Type{Array_64_Culonglong}) = begin  # /home/shindo/local-lemon/.julia/v0.4/Clang/src/wrap_c.jl, line 266:
        Array_64_Culonglong(fill(zero(Culonglong),64)...)
    end

typealias curandDirectionVectors64_t Array_64_Culonglong
typealias curandGenerator_st Void
typealias curandGenerator_t Ptr{curandGenerator_st}
typealias curandDistribution_st Cdouble
typealias curandDistribution_t Ptr{curandDistribution_st}
typealias curandDistributionShift_st Void
typealias curandDistributionShift_t Ptr{curandDistributionShift_st}
typealias curandDistributionM2Shift_st Void
typealias curandDistributionM2Shift_t Ptr{curandDistributionM2Shift_st}
typealias curandHistogramM2_st Void
typealias curandHistogramM2_t Ptr{curandHistogramM2_st}
typealias curandHistogramM2K_st UInt32
typealias curandHistogramM2K_t Ptr{curandHistogramM2K_st}
typealias curandHistogramM2V_st curandDistribution_st
typealias curandHistogramM2V_t Ptr{curandHistogramM2V_st}
typealias curandDiscreteDistribution_st Void
typealias curandDiscreteDistribution_t Ptr{curandDiscreteDistribution_st}

# begin enum curandMethod
typealias curandMethod UInt32
const CURAND_CHOOSE_BEST = (UInt32)(0)
const CURAND_ITR = (UInt32)(1)
const CURAND_KNUTH = (UInt32)(2)
const CURAND_HITR = (UInt32)(3)
const CURAND_M1 = (UInt32)(4)
const CURAND_M2 = (UInt32)(5)
const CURAND_BINARY_SEARCH = (UInt32)(6)
const CURAND_DISCRETE_GAUSS = (UInt32)(7)
const CURAND_REJECTION = (UInt32)(8)
const CURAND_DEVICE_API = (UInt32)(9)
const CURAND_FAST_REJECTION = (UInt32)(10)
const CURAND_3RD = (UInt32)(11)
const CURAND_DEFINITION = (UInt32)(12)
const CURAND_POISSON = (UInt32)(13)
# end enum curandMethod

typealias curandMethod_t curandMethod
