export batchsort

doc"""
    batchsort(x::Var, perm::Vector{Int})

Sort batched data according to the permutation vector of indices.
This is useful for sorting variable-length mini-batch data in e.g. descending order.

# 👉 Example
```julia
T = Float32
x = Var(rand(T,100,10))
perm = sortperm(x.batchdims, rev=true)
y = batchsort(x, perm)
```
"""
function batchsort(x::Var, perm::Vector{Int})
    issorted(perm) && return x
    s = unsafe_split(x.data, x.batchdims)
    y = cat(ndims(x), s[perm]...)
    Var(y, x.batchdims[perm], batchsort, (x,perm))
end

function addgrad!(y::Var, ::typeof(batchsort), x::Var, perm::Vector{Int})
    if !isvoid(x.grad)
        s = unsafe_split(y.grad, y.batchdims)
        gy = cat(ndims(y), s[perm]...)
        BLAS.axpy!(1, gy, x.grad)
    end
end
