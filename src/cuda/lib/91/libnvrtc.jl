# Julia wrapper for header: /usr/local/cuda/include/nvrtc.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function nvrtcGetErrorString(result)
    ccall((:nvrtcGetErrorString, libnvrtc), Cstring, (nvrtcResult,), result)
end

function nvrtcVersion(major, minor)
    ccall((:nvrtcVersion, libnvrtc), nvrtcResult, (Ptr{Cint}, Ptr{Cint}), major, minor)
end

function nvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames)
    ccall((:nvrtcCreateProgram, libnvrtc), nvrtcResult, (Ptr{nvrtcProgram}, Cstring, Cstring, Cint, Ptr{Cstring}, Ptr{Cstring}), prog, src, name, numHeaders, headers, includeNames)
end

function nvrtcDestroyProgram(prog)
    ccall((:nvrtcDestroyProgram, libnvrtc), nvrtcResult, (Ptr{nvrtcProgram},), prog)
end

function nvrtcCompileProgram(prog, numOptions, options)
    ccall((:nvrtcCompileProgram, libnvrtc), nvrtcResult, (nvrtcProgram, Cint, Ptr{Cstring}), prog, numOptions, options)
end

function nvrtcGetPTXSize(prog, ptxSizeRet)
    ccall((:nvrtcGetPTXSize, libnvrtc), nvrtcResult, (nvrtcProgram, Ptr{Csize_t}), prog, ptxSizeRet)
end

function nvrtcGetPTX(prog, ptx)
    ccall((:nvrtcGetPTX, libnvrtc), nvrtcResult, (nvrtcProgram, Cstring), prog, ptx)
end

function nvrtcGetProgramLogSize(prog, logSizeRet)
    ccall((:nvrtcGetProgramLogSize, libnvrtc), nvrtcResult, (nvrtcProgram, Ptr{Csize_t}), prog, logSizeRet)
end

function nvrtcGetProgramLog(prog, log)
    ccall((:nvrtcGetProgramLog, libnvrtc), nvrtcResult, (nvrtcProgram, Cstring), prog, log)
end

function nvrtcAddNameExpression(prog, name_expression)
    ccall((:nvrtcAddNameExpression, libnvrtc), nvrtcResult, (nvrtcProgram, Cstring), prog, name_expression)
end

function nvrtcGetLoweredName(prog, name_expression, lowered_name)
    ccall((:nvrtcGetLoweredName, libnvrtc), nvrtcResult, (nvrtcProgram, Cstring, Ptr{Cstring}), prog, name_expression, lowered_name)
end
