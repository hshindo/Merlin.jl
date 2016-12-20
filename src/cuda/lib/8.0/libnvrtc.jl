# Julia wrapper for header: /home/shindo/local-lemon/cuda-8.0/include/nvrtc.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function nvrtcGetErrorString(result)
    ccall((:nvrtcGetErrorString,libnvrtc),Ptr{UInt8},(nvrtcResult,),result)
end

function nvrtcVersion(major,minor)
    check_nvrtcresult(ccall((:nvrtcVersion,libnvrtc),nvrtcResult,(Ptr{Cint},Ptr{Cint}),major,minor))
end

function nvrtcCreateProgram(prog,src,name,numHeaders,headers,includeNames)
    check_nvrtcresult(ccall((:nvrtcCreateProgram,libnvrtc),nvrtcResult,(Ptr{nvrtcProgram},Ptr{UInt8},Ptr{UInt8},Cint,Ptr{Ptr{UInt8}},Ptr{Ptr{UInt8}}),prog,src,name,numHeaders,headers,includeNames))
end

function nvrtcDestroyProgram(prog)
    check_nvrtcresult(ccall((:nvrtcDestroyProgram,libnvrtc),nvrtcResult,(Ptr{nvrtcProgram},),prog))
end

function nvrtcCompileProgram(prog,numOptions,options)
    check_nvrtcresult(ccall((:nvrtcCompileProgram,libnvrtc),nvrtcResult,(nvrtcProgram,Cint,Ptr{Ptr{UInt8}}),prog,numOptions,options))
end

function nvrtcGetPTXSize(prog,ptxSizeRet)
    check_nvrtcresult(ccall((:nvrtcGetPTXSize,libnvrtc),nvrtcResult,(nvrtcProgram,Ptr{Csize_t}),prog,ptxSizeRet))
end

function nvrtcGetPTX(prog,ptx)
    check_nvrtcresult(ccall((:nvrtcGetPTX,libnvrtc),nvrtcResult,(nvrtcProgram,Ptr{UInt8}),prog,ptx))
end

function nvrtcGetProgramLogSize(prog,logSizeRet)
    check_nvrtcresult(ccall((:nvrtcGetProgramLogSize,libnvrtc),nvrtcResult,(nvrtcProgram,Ptr{Csize_t}),prog,logSizeRet))
end

function nvrtcGetProgramLog(prog,log)
    check_nvrtcresult(ccall((:nvrtcGetProgramLog,libnvrtc),nvrtcResult,(nvrtcProgram,Ptr{UInt8}),prog,log))
end

function nvrtcAddNameExpression(prog,name_expression)
    check_nvrtcresult(ccall((:nvrtcAddNameExpression,libnvrtc),nvrtcResult,(nvrtcProgram,Ptr{UInt8}),prog,name_expression))
end

function nvrtcGetLoweredName(prog,name_expression,lowered_name)
    check_nvrtcresult(ccall((:nvrtcGetLoweredName,libnvrtc),nvrtcResult,(nvrtcProgram,Ptr{UInt8},Ptr{Ptr{UInt8}}),prog,name_expression,lowered_name))
end
