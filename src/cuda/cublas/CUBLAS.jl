module CUBLAS

using ..CUDA

include("define.jl")

macro cublas(f, rettypes, args...)
    f = get(DEFINE, f.value, f.value)
    quote
        status = ccall(($(QuoteNode(f)),CUDA.libcublas), Cint, $(esc(rettypes)), $(map(esc,args)...))
        if status != 0
            throw(ERROR_MESSAGE[status])
        end
    end
end

function version()
    h = gethandle()
    ref = Ref{Cint}()
    @cublas :cublasGetVersion (Ptr{Cvoid},Ptr{Cint}) h ref
    Int(ref[])
end

function cublasop(t::Char)
    t == 'N' && return Cint(0)
    t == 'T' && return Cint(1)
    t == 'C' && return Cint(2)
    throw("Unknown cublas operation: $(t).")
end

include("handle.jl")
include("level1.jl")
include("level2.jl")
include("level3.jl")
include("extension.jl")

end
