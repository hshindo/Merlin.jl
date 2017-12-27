module CUBLAS

import ..CUDA: ndevices, getdevice, setdevice, CuArray, CuMatrix, CuVector, CuVecOrMat

if is_windows()
    const libcublas = Libdl.find_library(["cublas64_91","cublas64_90","cublas64_80","cublas64_75"])
else
    const libcublas = Libdl.find_library(["libcublas"])
end
isempty(libcublas) && warn("CUBLAS library cannot be found.")

@enum(CUBLAS_status,
    CUBLAS_STATUS_SUCCESS = Cint(0),
    CUBLAS_STATUS_NOT_INITIALIZED = Cint(1),
    CUBLAS_STATUS_ALLOC_FAILED = Cint(3),
    CUBLAS_STATUS_INVALID_VALUE = Cint(7),
    CUBLAS_STATUS_ARCH_MISMATCH = Cint(8),
    CUBLAS_STATUS_MAPPING_ERROR = Cint(11),
    CUBLAS_STATUS_EXECUTION_FAILED = Cint(13),
    CUBLAS_STATUS_INTERNAL_ERROR = Cint(14),
    CUBLAS_STATUS_NOT_SUPPORTED = Cint(15),
    CUBLAS_STATUS_LICENSE_ERROR = Cint(16))

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

include("define.jl")

macro apicall(f, args...)
    f = get(define, f.args[1], f.args[1])
    quote
        status = ccall(($(QuoteNode(f)),libcublas), Cint, $(map(esc,args)...))
        if status != 0
            Base.show_backtrace(STDOUT, backtrace())
            throw(errorstring(status))
        end
    end
end

const cublasHandle_t = Ptr{Void}
const Handles = cublasHandle_t[]

handle(x::CuArray) = Handles[getdevice(x)+1]

function init()
    empty!(Handles)
    for dev = 0:ndevices()-1
        setdevice(dev)
        ref = Ref{cublasHandle_t}()
        @apicall :cublasCreate (cublasHandle_t,) ref
        h = ref[]
        atexit() do
            @apicall :cublasDestroy (cublasHandle_t,) h
        end
        push!(Handles, h)
    end
    setdevice(0)
end
init()

const API_VERSION = begin
    ref = Ref{Cint}()
    @apicall :cublasGetVersion (cublasHandle_t,Ptr{Cint}) Handles[1] ref
    Int(ref[])
end
info("CUBLAS API $API_VERSION")

function cublasop(t::Char)
    t == 'N' && return Cint(0)
    t == 'T' && return Cint(1)
    t == 'C' && return Cint(2)
    throw("Unknown cublas operation: $(t).")
end

#include("level1.jl")
include("level2.jl")
include("level3.jl")

end
