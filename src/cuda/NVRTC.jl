module NVRTC

if is_windows()
    const libnvrtc = Libdl.find_library(["nvrtc64_91","nvrtc64_90","nvrtc64_80","nvrtc64_75"])
else
    const libnvrtc = Libdl.find_library(["libnvrtc"])
end
isempty(libnvrtc) && warn("NVRTC library cannot be found.")

const nvrtcProgram = Ptr{Void}

macro apicall(f, args...)
    quote
        result = ccall(($f,libnvrtc), Cint, $(map(esc,args)...))
        if result != 0
            Base.show_backtrace(STDOUT, backtrace())
            p = ccall((:nvrtcGetErrorString,libnvrtc), Cstring, (Cint,), result)
            throw(unsafe_string(p))
        end
    end
end

const API_VERSION = begin
    ref_major = Ref{Cint}()
    ref_minor = Ref{Cint}()
    @apicall :nvrtcVersion (Ptr{Cint},Ptr{Cint}) ref_major ref_minor
    major = Int(ref_major[])
    minor = Int(ref_minor[])
    1000major + 10minor
end
info("NVRTC API $API_VERSION")

function compile(code::String; headers=(), include_names=(), options=[])
    ref = Ref{nvrtcProgram}()
    headers = Ptr{UInt8}[pointer(h) for h in headers]
    include_names = Ptr{UInt8}[pointer(n) for n in include_names]
    @apicall :nvrtcCreateProgram (Ptr{nvrtcProgram},Cstring,Cstring,Cint,Ptr{Ptr{UInt8}},Ptr{Ptr{UInt8}}) ref code C_NULL length(headers) headers include_names
    prog = ref[]

    options = Ptr{UInt8}[pointer(o) for o in options]
    try
        @apicall :nvrtcCompileProgram (nvrtcProgram,Cint,Ptr{Ptr{UInt8}}) prog length(options) options
    catch
        ref = Ref{Csize_t}()
        @apicall :nvrtcGetProgramLogSize (nvrtcProgram,Ptr{Csize_t}) prog ref
        log = Array{UInt8}(Int(ref[]))
        @apicall :nvrtcGetProgramLog (nvrtcProgram,Ptr{UInt8}) prog log
        println()
        println("Error log:")
        println(String(log))
        throw("NVRTC compile failed.")
    end

    ref = Ref{Csize_t}()
    @apicall :nvrtcGetPTXSize (nvrtcProgram,Ptr{Csize_t}) prog ref
    ptxsize = ref[]

    ptx = Array{UInt8}(ptxsize)
    @apicall :nvrtcGetPTX (nvrtcProgram,Ptr{UInt8}) prog ptx
    @apicall :nvrtcDestroyProgram (Ptr{nvrtcProgram},) Ref{nvrtcProgram}(prog)

    String(ptx)
end

end
