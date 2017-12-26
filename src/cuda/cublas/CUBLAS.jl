module CUBLAS

import Base.LinAlg.BLAS: axpy!, gemv, gemv!, gemm, gemm!

if is_windows()
    const libcublas = Libdl.find_library(["cublas64_91","cublas64_90","cublas64_80","cublas64_75"])
else
    const libcublas = Libdl.find_library(["libcublas"])
end
isempty(libcublas) && warn("CUBLAS library cannot be found.")

function cublasCreate_v2(handle)
    check_cublasstatus(ccall((:cublasCreate_v2,libcublas),cublasStatus_t,(Ptr{cublasHandle_t},),handle))
end

function cublasDestroy_v2(handle)
    check_cublasstatus(ccall((:cublasDestroy_v2,libcublas),cublasStatus_t,(cublasHandle_t,),handle))
end

const cublasHandle_t = Ptr{Void}

const handles = Ptr{Void}[]
handle(x::CuArray) = handles[device(x)+1]
atexit(() -> foreach(cublasDestroy,handles))

include("define.jl")

macro apicall(f, args...)
    f = get(define, f.args[1], f.args[1])
    quote
        status = ccall(($(QuoteNode(f)),libcublas), Cint, $(map(esc,args)...))
        if status != 0
            Base.show_backtrace(STDOUT, backtrace())
            ref = Ref{Cstring}()
            ccall((:cuGetErrorString,libcuda), Cint, (Cint,Ptr{Cstring}), status, ref)
            throw(unsafe_string(ref[]))
        end
    end
end

function init()
    isempty(handles) || throw("Handle is not empty.")
    for dev = ndevices()-1:-1:0
        setdevice(dev)
        ref = Ref{Void}()
        @apicall :cublasCreate (Ptr{Void},) ref
        push!(handles, ref[])
    end
end
init()

function version()
    ref = Ref{Cint}()
    @apicall :cublasGetVersion (cublasHandle_t,Ptr{Cint}) handle ref
    Int(ref[])
end
const API_VERSION = version()
info("CUBLAS API $API_VERSION")

function errorstring(status)
    status == CUBLAS_STATUS_SUCCESS && return "SUCCESS"
    status == CUBLAS_STATUS_NOT_INITIALIZED && return "NOT_INITIALIZED"
    status == CUBLAS_STATUS_ALLOC_FAILED && return "ALLOC_FAILED"
    status == CUBLAS_STATUS_INVALID_VALUE && return "INVALID_VALUE"
    status == CUBLAS_STATUS_ARCH_MISMATCH && return "ARCH_MISMATCH"
    status == CUBLAS_STATUS_MAPPING_ERROR && return "MAPPING_ERROR"
    status == CUBLAS_STATUS_EXECUTION_FAILED && return "EXECUTION_FAILED"
    status == CUBLAS_STATUS_INTERNAL_ERROR && return "INTERNAL_ERROR"
    status == CUBLAS_STATUS_NOT_SUPPORTED && return "NOT_SUPPORTED"
    status == CUBLAS_STATUS_LICENSE_ERROR && return "LICENSE_ERROR"
    throw("UNKNOWN ERROR")
end

function cublasop(t::Char)
    t == 'N' && return CUBLAS_OP_N
    t == 'T' && return CUBLAS_OP_T
    t == 'C' && return CUBLAS_OP_C
    throw("Unknown cublas operation: $(t).")
end

end
