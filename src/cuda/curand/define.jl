const CURAND_STATUS_SUCCESS = 0
const CURAND_STATUS_VERSION_MISMATCH = 100
const CURAND_STATUS_NOT_INITIALIZED = 101
const CURAND_STATUS_ALLOCATION_FAILED = 102
const CURAND_STATUS_TYPE_ERROR = 103
const CURAND_STATUS_OUT_OF_RANGE = 104
const CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105
const CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
const CURAND_STATUS_LAUNCH_FAILURE = 201
const CURAND_STATUS_PREEXISTING_FAILURE = 202
const CURAND_STATUS_INITIALIZATION_FAILED = 203
const CURAND_STATUS_ARCH_MISMATCH = 204
const CURAND_STATUS_INTERNAL_ERROR = 999

const ERROR_MESSAGES = Dict(
   CURAND_STATUS_SUCCESS => "No errors",
   CURAND_STATUS_VERSION_MISMATCH => "Header file and linked library version do not match",
   CURAND_STATUS_NOT_INITIALIZED => "Generator not initialized",
   CURAND_STATUS_ALLOCATION_FAILED => "Memory allocation failed",
   CURAND_STATUS_TYPE_ERROR => "Generator is wrong type",
   CURAND_STATUS_OUT_OF_RANGE => "Argument out of range",
   CURAND_STATUS_LENGTH_NOT_MULTIPLE => "Length requested is not a multple of dimension",
   CURAND_STATUS_DOUBLE_PRECISION_REQUIRED => "GPU does not have double precision required by MRG32k3a",
   CURAND_STATUS_LAUNCH_FAILURE => "Kernel launch failure",
   CURAND_STATUS_PREEXISTING_FAILURE => "Preexisting failure on library entry",
   CURAND_STATUS_INITIALIZATION_FAILED => "Initialization of CUDA failed",
   CURAND_STATUS_ARCH_MISMATCH => "Architecture mismatch, GPU does not support requested feature",
   CURAND_STATUS_INTERNAL_ERROR => "Internal library error"
)

const CURAND_RNG_TEST = 0
const CURAND_RNG_PSEUDO_DEFAULT = 100 # Default pseudorandom generator
const CURAND_RNG_PSEUDO_XORWOW = 101 # XORWOW pseudorandom generator
const CURAND_RNG_PSEUDO_MRG32K3A = 121 # MRG32k3a pseudorandom generator
const CURAND_RNG_PSEUDO_MTGP32 = 141 # Mersenne Twister MTGP32 pseudorandom generator
const CURAND_RNG_PSEUDO_MT19937 = 142 # Mersenne Twister MT19937 pseudorandom generator
const CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161 # PHILOX-4x32-10 pseudorandom generator
const CURAND_RNG_QUASI_DEFAULT = 200 # Default quasirandom generator
const CURAND_RNG_QUASI_SOBOL32 = 201 # Sobol32 quasirandom generator
const CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202 # Scrambled Sobol32 quasirandom generator
const CURAND_RNG_QUASI_SOBOL64 = 203 # Sobol64 quasirandom generator
const CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204 # Scrambled Sobol64 quasirandom generator

#=
enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100, ///< Best ordering for pseudorandom results
    CURAND_ORDERING_PSEUDO_DEFAULT = 101, ///< Specific default 4096 thread sequence for pseudorandom results
    CURAND_ORDERING_PSEUDO_SEEDED = 102, ///< Specific seeding pattern for fast lower quality pseudorandom results
    CURAND_ORDERING_QUASI_DEFAULT = 201 ///< Specific n-dimensional ordering for quasirandom results
};

enum curandDirectionVectorSet {
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101, ///< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102, ///< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103, ///< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104 ///< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
};
=#
