type Var{T}
    data::T
    grad
    f
    df
    args::Vector
    sess
end

Var() = Var(nothing, nothing)
Var(data) = Var(data, nothing, nothing, [])
Var(f, args::Vector) = Var(nothing, nothing, f, args)

function Var{T}(v::Var, ::Type{T}, dims)
    data = alloc(v.sess, T, dims)
    Var(data, nothing, nothing, nothing, [], v.sess)
end

Base.similar(v::Var) = Var(v, eltype(v), size(v))

function Base.Array(mp::MemoryPool, T::Type, dims::Tuple{Vararg{Int}})
    len = prod(dims) * sizeof(T)
    @assert len > 0

    count = mp.length - mp.index + 1
    if count < len
        while mp.length < mp.index+len-1
            mp.length *= 2
        end
        mp.array = Array(Bool, mp.length)
        mp.index = 1
    end

    p = pointer(mp.array, mp.index)
    mp.index += len
    unsafe_wrap(Array, Ptr{T}(p), dims)
end
