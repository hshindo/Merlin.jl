const CUBLAS_STATUS_SUCCESS = 0
const CUBLAS_STATUS_NOT_INITIALIZED = 1
const CUBLAS_STATUS_ALLOC_FAILED = 3
const CUBLAS_STATUS_INVALID_VALUE = 7
const CUBLAS_STATUS_ARCH_MISMATCH = 8
const CUBLAS_STATUS_MAPPING_ERROR = 11
const CUBLAS_STATUS_EXECUTION_FAILED = 13
const CUBLAS_STATUS_INTERNAL_ERROR = 14
const CUBLAS_STATUS_NOT_SUPPORTED = 15
const CUBLAS_STATUS_LICENSE_ERROR = 16

const ERROR_MESSAGE = Dict(
    CUBLAS_STATUS_SUCCESS => "SUCCESS",
    CUBLAS_STATUS_NOT_INITIALIZED => "NOT_INITIALIZED",
    CUBLAS_STATUS_ALLOC_FAILED => "ALLOC_FAILED",
    CUBLAS_STATUS_INVALID_VALUE => "INVALID_VALUE",
    CUBLAS_STATUS_ARCH_MISMATCH => "ARCH_MISMATCH",
    CUBLAS_STATUS_MAPPING_ERROR => "MAPPING_ERROR",
    CUBLAS_STATUS_EXECUTION_FAILED => "EXECUTION_FAILED",
    CUBLAS_STATUS_INTERNAL_ERROR => "INTERNAL_ERROR",
    CUBLAS_STATUS_NOT_SUPPORTED => "NOT_SUPPORTED",
    CUBLAS_STATUS_LICENSE_ERROR => "LICENSE_ERROR"
)

const DEFINE = Dict{Symbol,Symbol}()

DEFINE[:cublasCreate] = :cublasCreate_v2
DEFINE[:cublasDestroy] = :cublasDestroy_v2
DEFINE[:cublasGetVersion] = :cublasGetVersion_v2
DEFINE[:cublasSetStream] = :cublasSetStream_v2
DEFINE[:cublasGetStream] = :cublasGetStream_v2
DEFINE[:cublasGetPointerMode] = :cublasGetPointerMode_v2
DEFINE[:cublasSetPointerMode] = :cublasSetPointerMode_v2

# Blas3 RoutinesS

DEFINE[:cublasSnrm2] = :cublasSnrm2_v2
DEFINE[:cublasDnrm2] = :cublasDnrm2_v2
DEFINE[:cublasScnrm2] = :cublasScnrm2_v2
DEFINE[:cublasDznrm2] = :cublasDznrm2_v2

DEFINE[:cublasSdot] = :cublasSdot_v2
DEFINE[:cublasDdot] = :cublasDdot_v2
DEFINE[:cublasCdotu] = :cublasCdotu_v2
DEFINE[:cublasCdotc] = :cublasCdotc_v2
DEFINE[:cublasZdotu] = :cublasZdotu_v2
DEFINE[:cublasZdotc] = :cublasZdotc_v2

DEFINE[:cublasSscal] = :cublasSscal_v2
DEFINE[:cublasDscal] = :cublasDscal_v2
DEFINE[:cublasCscal] = :cublasCscal_v2
DEFINE[:cublasCsscal] = :cublasCsscal_v2
DEFINE[:cublasZscal] = :cublasZscal_v2
DEFINE[:cublasZdscal] = :cublasZdscal_v2

DEFINE[:cublasSaxpy] = :cublasSaxpy_v2
DEFINE[:cublasDaxpy] = :cublasDaxpy_v2
DEFINE[:cublasCaxpy] = :cublasCaxpy_v2
DEFINE[:cublasZaxpy] = :cublasZaxpy_v2

DEFINE[:cublasScopy] = :cublasScopy_v2
DEFINE[:cublasDcopy] = :cublasDcopy_v2
DEFINE[:cublasCcopy] = :cublasCcopy_v2
DEFINE[:cublasZcopy] = :cublasZcopy_v2

DEFINE[:cublasSswap] = :cublasSswap_v2
DEFINE[:cublasDswap] = :cublasDswap_v2
DEFINE[:cublasCswap] = :cublasCswap_v2
DEFINE[:cublasZswap] = :cublasZswap_v2

DEFINE[:cublasIsamax] = :cublasIsamax_v2
DEFINE[:cublasIdamax] = :cublasIdamax_v2
DEFINE[:cublasIcamax] = :cublasIcamax_v2
DEFINE[:cublasIzamax] = :cublasIzamax_v2

DEFINE[:cublasIsamin] = :cublasIsamin_v2
DEFINE[:cublasIdamin] = :cublasIdamin_v2
DEFINE[:cublasIcamin] = :cublasIcamin_v2
DEFINE[:cublasIzamin] = :cublasIzamin_v2

DEFINE[:cublasSasum] = :cublasSasum_v2
DEFINE[:cublasDasum] = :cublasDasum_v2
DEFINE[:cublasScasum] = :cublasScasum_v2
DEFINE[:cublasDzasum] = :cublasDzasum_v2

DEFINE[:cublasSrot] = :cublasSrot_v2
DEFINE[:cublasDrot] = :cublasDrot_v2
DEFINE[:cublasCrot] = :cublasCrot_v2
DEFINE[:cublasCsrot] = :cublasCsrot_v2
DEFINE[:cublasZrot] = :cublasZrot_v2
DEFINE[:cublasZdrot] = :cublasZdrot_v2

DEFINE[:cublasSrotg] = :cublasSrotg_v2
DEFINE[:cublasDrotg] = :cublasDrotg_v2
DEFINE[:cublasCrotg] = :cublasCrotg_v2
DEFINE[:cublasZrotg] = :cublasZrotg_v2

DEFINE[:cublasSrotm] = :cublasSrotm_v2
DEFINE[:cublasDrotm] = :cublasDrotm_v2

DEFINE[:cublasSrotmg] = :cublasSrotmg_v2
DEFINE[:cublasDrotmg] = :cublasDrotmg_v2


# Blas2 Routines

DEFINE[:cublasSgemv] = :cublasSgemv_v2
DEFINE[:cublasDgemv] = :cublasDgemv_v2
DEFINE[:cublasCgemv] = :cublasCgemv_v2
DEFINE[:cublasZgemv] = :cublasZgemv_v2

DEFINE[:cublasSgbmv] = :cublasSgbmv_v2
DEFINE[:cublasDgbmv] = :cublasDgbmv_v2
DEFINE[:cublasCgbmv] = :cublasCgbmv_v2
DEFINE[:cublasZgbmv] = :cublasZgbmv_v2

DEFINE[:cublasStrmv] = :cublasStrmv_v2
DEFINE[:cublasDtrmv] = :cublasDtrmv_v2
DEFINE[:cublasCtrmv] = :cublasCtrmv_v2
DEFINE[:cublasZtrmv] = :cublasZtrmv_v2

DEFINE[:cublasStbmv] = :cublasStbmv_v2
DEFINE[:cublasDtbmv] = :cublasDtbmv_v2
DEFINE[:cublasCtbmv] = :cublasCtbmv_v2
DEFINE[:cublasZtbmv] = :cublasZtbmv_v2

DEFINE[:cublasStpmv] = :cublasStpmv_v2
DEFINE[:cublasDtpmv] = :cublasDtpmv_v2
DEFINE[:cublasCtpmv] = :cublasCtpmv_v2
DEFINE[:cublasZtpmv] = :cublasZtpmv_v2

DEFINE[:cublasStrsv] = :cublasStrsv_v2
DEFINE[:cublasDtrsv] = :cublasDtrsv_v2
DEFINE[:cublasCtrsv] = :cublasCtrsv_v2
DEFINE[:cublasZtrsv] = :cublasZtrsv_v2

DEFINE[:cublasStpsv] = :cublasStpsv_v2
DEFINE[:cublasDtpsv] = :cublasDtpsv_v2
DEFINE[:cublasCtpsv] = :cublasCtpsv_v2
DEFINE[:cublasZtpsv] = :cublasZtpsv_v2

DEFINE[:cublasStbsv] = :cublasStbsv_v2
DEFINE[:cublasDtbsv] = :cublasDtbsv_v2
DEFINE[:cublasCtbsv] = :cublasCtbsv_v2
DEFINE[:cublasZtbsv] = :cublasZtbsv_v2

DEFINE[:cublasSsymv] = :cublasSsymv_v2
DEFINE[:cublasDsymv] = :cublasDsymv_v2
DEFINE[:cublasCsymv] = :cublasCsymv_v2
DEFINE[:cublasZsymv] = :cublasZsymv_v2
DEFINE[:cublasChemv] = :cublasChemv_v2
DEFINE[:cublasZhemv] = :cublasZhemv_v2

DEFINE[:cublasSsbmv] = :cublasSsbmv_v2
DEFINE[:cublasDsbmv] = :cublasDsbmv_v2
DEFINE[:cublasChbmv] = :cublasChbmv_v2
DEFINE[:cublasZhbmv] = :cublasZhbmv_v2

DEFINE[:cublasSspmv] = :cublasSspmv_v2
DEFINE[:cublasDspmv] = :cublasDspmv_v2
DEFINE[:cublasChpmv] = :cublasChpmv_v2
DEFINE[:cublasZhpmv] = :cublasZhpmv_v2

DEFINE[:cublasSger] = :cublasSger_v2
DEFINE[:cublasDger] = :cublasDger_v2
DEFINE[:cublasCgeru] = :cublasCgeru_v2
DEFINE[:cublasCgerc] = :cublasCgerc_v2
DEFINE[:cublasZgeru] = :cublasZgeru_v2
DEFINE[:cublasZgerc] = :cublasZgerc_v2

DEFINE[:cublasSsyr] = :cublasSsyr_v2
DEFINE[:cublasDsyr] = :cublasDsyr_v2
DEFINE[:cublasCsyr] = :cublasCsyr_v2
DEFINE[:cublasZsyr] = :cublasZsyr_v2
DEFINE[:cublasCher] = :cublasCher_v2
DEFINE[:cublasZher] = :cublasZher_v2

DEFINE[:cublasSspr] = :cublasSspr_v2
DEFINE[:cublasDspr] = :cublasDspr_v2
DEFINE[:cublasChpr] = :cublasChpr_v2
DEFINE[:cublasZhpr] = :cublasZhpr_v2

DEFINE[:cublasSsyr2] = :cublasSsyr2_v2
DEFINE[:cublasDsyr2] = :cublasDsyr2_v2
DEFINE[:cublasCsyr2] = :cublasCsyr2_v2
DEFINE[:cublasZsyr2] = :cublasZsyr2_v2
DEFINE[:cublasCher2] = :cublasCher2_v2
DEFINE[:cublasZher2] = :cublasZher2_v2

DEFINE[:cublasSspr2] = :cublasSspr2_v2
DEFINE[:cublasDspr2] = :cublasDspr2_v2
DEFINE[:cublasChpr2] = :cublasChpr2_v2
DEFINE[:cublasZhpr2] = :cublasZhpr2_v2

# Blas3 Routines

DEFINE[:cublasSgemm] = :cublasSgemm_v2
DEFINE[:cublasDgemm] = :cublasDgemm_v2
DEFINE[:cublasCgemm] = :cublasCgemm_v2
DEFINE[:cublasZgemm] = :cublasZgemm_v2

DEFINE[:cublasSsyrk] = :cublasSsyrk_v2
DEFINE[:cublasDsyrk] = :cublasDsyrk_v2
DEFINE[:cublasCsyrk] = :cublasCsyrk_v2
DEFINE[:cublasZsyrk] = :cublasZsyrk_v2
DEFINE[:cublasCherk] = :cublasCherk_v2
DEFINE[:cublasZherk] = :cublasZherk_v2

DEFINE[:cublasSsyr2k] = :cublasSsyr2k_v2
DEFINE[:cublasDsyr2k] = :cublasDsyr2k_v2
DEFINE[:cublasCsyr2k] = :cublasCsyr2k_v2
DEFINE[:cublasZsyr2k] = :cublasZsyr2k_v2
DEFINE[:cublasCher2k] = :cublasCher2k_v2
DEFINE[:cublasZher2k] = :cublasZher2k_v2

DEFINE[:cublasSsymm] = :cublasSsymm_v2
DEFINE[:cublasDsymm] = :cublasDsymm_v2
DEFINE[:cublasCsymm] = :cublasCsymm_v2
DEFINE[:cublasZsymm] = :cublasZsymm_v2
DEFINE[:cublasChemm] = :cublasChemm_v2
DEFINE[:cublasZhemm] = :cublasZhemm_v2

DEFINE[:cublasStrsm] = :cublasStrsm_v2
DEFINE[:cublasDtrsm] = :cublasDtrsm_v2
DEFINE[:cublasCtrsm] = :cublasCtrsm_v2
DEFINE[:cublasZtrsm] = :cublasZtrsm_v2

DEFINE[:cublasStrmm] = :cublasStrmm_v2
DEFINE[:cublasDtrmm] = :cublasDtrmm_v2
DEFINE[:cublasCtrmm] = :cublasCtrmm_v2
DEFINE[:cublasZtrmm] = :cublasZtrmm_v2
