const MALLOC = Ref{Any}(CUDAMalloc())

function malloc(bytesize::Int)
    @assert bytesize >= 0
    bytesize == 0 && return MemBlock(C_NULL,0)
    MALLOC[](bytesize)
end

function setmalloc(f::Function, m)
    _m = MALLOC[]
    MALLOC[] = MALLOC[name]
    f()
    MALLOC[] = _m
end
