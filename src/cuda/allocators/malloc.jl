const MALLOC = Ref{Any}(CUDAMalloc())

function malloc(::Type{T}, size::Int) where T
    @assert size >= 0
    ptr, size = MALLOC[](bytesize)
    size == 0 ? CuPtr(Ptr{T}(C_NULL),0,-1,false) : MALLOC[](T,size)
end

function setmalloc(f::Function, m)
    _m = MALLOC[]
    MALLOC[] = MALLOC[name]
    f()
    MALLOC[] = _m
end
