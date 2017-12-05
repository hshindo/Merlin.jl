import Base: reshape, merge, repeat

doc"""
    reshape(x, dims::Tuple, batchdims::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,5))
y = reshape(x, (2,5), [2,3])
```
"""
function reshape(x::Var, dims::Tuple, batchdims::Vector{Int})
    y = reshape(x.data, dims)
    sum(batchdims) == size(y,ndims(y)) || throw("Invalid batchdims: $(size(y)) and $batchdims")
    Var(y, batchdims, reshape, (x,))
end

reshape(x::Node, dims::Tuple, batchdims::Vector{Int}; name="") = Node(reshape, (x,dims), name)

function addgrad!(y::Var, ::typeof(reshape), x::Var)
    if !isvoid(x.grad)
        T = eltype(x)
        BLAS.axpy!(T(1), y.grad, x.grad)
    end
end

doc"""
    merge(x::Var)
"""
merge(x::Var) = reshape(x, size(x), [sum(x.batchdims)])

merge(x::Node; name="") = Node(merge, (x,), name)

doc"""
    repeat(x::Var, nums::Int...)
"""
function repeat(x::Var, nums::Int...)
    y = similar(x.data, ntuple(i -> size(x,i)*nums[i], length(nums)))
    s = unsafe_split(x.data, x.batchdims)
    for i = 1:length(s)
        y[] = s[i]
    end

    y = repmat(x.data, m, n)
    Var(y, x.batchdims, repmat, (x,m,n))
end

repmat(x::Node, m::Int, n::Int=1, name="") = Node(repmat, (x,m,n), name)

#=
function promote_size(x::Var)
    dims = x.batchdims
    all(d -> d == dims[1], dims) || return x
    s = Base.front(size(x))..., dims[1], size(x,ndims(x))÷dims[1]
    reshape(x, s, [s[end]])
end

function promote_size(x::Array, batchdims::Vector{Int})
    dims = batchdims
    all(d -> d == dims[1], dims) || return x
    s = Base.front(size(x))..., dims[1], size(x,ndims(x))÷dims[1]
    reshape(x, s)
end
=#
