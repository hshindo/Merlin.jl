module NVRTC

using ..CUDA

macro nvrtc(f, args...)
    quote
        result = ccall(($f,CUDA.libnvrtc), Cint, $(map(esc,args)...))
        if result != 0
            Base.show_backtrace(STDOUT, backtrace())
            p = ccall((:nvrtcGetErrorString,libnvrtc), Cstring, (Cint,), result)
            throw(unsafe_string(p))
        end
    end
end

function version()
    ref_major = Ref{Cint}()
    ref_minor = Ref{Cint}()
    @nvrtc :nvrtcVersion (Ptr{Cint},Ptr{Cint}) ref_major ref_minor
    1000 * Int(ref_major[]) + 10 * Int(ref_minor[])
end

function compile(code::String; headers=[], include_names=[], options=[])
    #options = ["-lineinfo", "-G"]

    ref = Ref{Ptr{Cvoid}}()
    headers = Ptr{UInt8}[pointer(h) for h in headers]
    include_names = Ptr{UInt8}[pointer(n) for n in include_names]
    @nvrtc :nvrtcCreateProgram (Ptr{Ptr{Cvoid}},Cstring,Cstring,Cint,Ptr{Ptr{UInt8}},Ptr{Ptr{UInt8}}) ref code C_NULL length(headers) headers include_names
    prog = ref[]

    options = Ptr{UInt8}[pointer(o) for o in options]
    try
        @nvrtc :nvrtcCompileProgram (Ptr{Cvoid},Cint,Ptr{Ptr{UInt8}}) prog length(options) options
    catch
        ref = Ref{Csize_t}()
        @nvrtc :nvrtcGetProgramLogSize (Ptr{Cvoid},Ptr{Csize_t}) prog ref
        log = Array{UInt8}(undef, Int(ref[]))
        @nvrtc :nvrtcGetProgramLog (Ptr{Cvoid},Ptr{UInt8}) prog log
        println()
        println("Error log:")
        println(String(log))
        throw("NVRTC compile failed.")
    end

    ref = Ref{Csize_t}()
    @nvrtc :nvrtcGetPTXSize (Ptr{Cvoid},Ptr{Csize_t}) prog ref
    ptxsize = ref[]

    ptx = Array{UInt8}(undef, ptxsize)
    @nvrtc :nvrtcGetPTX (Ptr{Cvoid},Ptr{UInt8}) prog ptx
    @nvrtc :nvrtcDestroyProgram (Ptr{Ptr{Cvoid}},) Ref{Ptr{Cvoid}}(prog)

    String(ptx)
end

end
