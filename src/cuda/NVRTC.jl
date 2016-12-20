module NVRTC

using ..CUDA

include("lib/$(CUDA.major).$(CUDA.minor)/libnvrtc.jl")
include("lib/$(CUDA.major).$(CUDA.minor)/libnvrtc_types.jl")

if is_windows()
    const libnvrtc = Libdl.find_library(["nvrtc64_80","nvrtc64_75"])
else
    const libnvrtc = Libdl.find_library(["libnvrtc"])
end
isempty(libnvrtc) && error("NVRTC library cannot be found.")

function check_nvrtcresult(status)
    status == NVRTC_SUCCESS && return nothing
    warn("NVRTC error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    throw(unsafe_string(nvrtcGetErrorString(status)))
end

function getlog(prog::Ptr{Void})
    logsize = Csize_t[0]
    nvrtcGetProgramLogSize(prog, logsize)
    log = Array(UInt8, logsize[1])
    nvrtcGetProgramLog(prog, log)
    unsafe_string(log)
end

function compile(code::String; headers=(), include_names=())
    ref = Ptr{Void}[0]
    headers = Ptr{Void}[pointer(h) for h in headers]
    include_names = Ptr{Void}[pointer(n) for n in include_names]
    nvrtcCreateProgram(ref, code, C_NULL, length(headers), headers, include_names)

    prog = ref[1]
    options = ["--gpu-architecture=compute_30"]
    nvrtcCompileProgram(prog, length(options), options)
    ptxsize = Csize_t[0]
    nvrtcGetPTXSize(prog, ptxsize)
    ptx = Array(UInt8, ptxsize[1])
    nvrtcGetPTX(prog, ptx)
    #log = getlog(prog)
    #throw(log)
    nvrtcDestroyProgram(ref)
    String(ptx)
end

end
