export dropout

doc"""
    dropout(x::Var, droprate::Float64)

Drops elements randomly with probability ``droprate`` and scales the other elements by factor ``1 / (1 - droprate)``.
"""
function dropout(x::Var, droprate::Float64)
    droprate == 0.0 && return x
    istraining() || return x
    ydata, work = dropout(x.data, droprate)
    Var(ydata, ∇dropout!, (x,droprate,work))
end
dropout(x::Node, droprate) = Node(dropout, (x,droprate))

function dropout(x::Array{T}, droprate::Float64) where T
    work = rand(T, length(x))
    scale = T(1 / (1-droprate))
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = work[i] <= droprate ? T(0) : scale*x[i]
    end
    y, work
end

dropout(x::CuArray, droprate) = CUDNN.dropout(x, droprate)

function ∇dropout!(y::Var, x::Var, droprate::Float64, work)
    isnothing(x.grad) && return
    ∇dropout!(y.grad, x.grad, droprate, work)
end

function ∇dropout!(gy::Array{T}, gx::Array{T}, droprate::Float64, work::Vector{T}) where T
    scale = T(1 / (1-droprate))
    @inbounds for i = 1:length(gx)
        gx[i] += work[i] <= droprate ? T(0) : scale*gy[i]
    end
end

∇dropout!(gy::CuArray, gx::CuArray, droprate, dropdesc) = CUDNN.∇dropout!(gy, gx, dropdesc)

export LockedDropout
mutable struct LockedDropout
    droprate
    mask
end

LockedDropout(droprate) = LockedDropout(droprate, nothing)

function (f::LockedDropout)(x::Var)
    f.droprate == 0.0 && return x
    istraining() || return x
    if f.mask == nothing
        f.mask = dropout_mask(x.data, f.droprate)
    end
    Var(f.mask) .* x
end

@generated function dropout_mask(x::CuArray{T}, droprate::Float64) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void dropout_dim($Ct *y, $Ct droprate, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        y[idx] = y[idx] <= droprate ? 0 : 1 / (1-droprate);
    }
    """)
    quote
        @assert 0.0 < droprate < 1.0
        y = curand(T, size(x))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), T(droprate), length(y))
        y
    end
end

export LockedDropoutDim
mutable struct LockedDropoutDim
    mask
end

LockedDropoutDim() = LockedDropoutDim(nothing)

function (f::LockedDropoutDim)(x::Var, dim::Int, droprate::Float64, scaling::Bool)
    droprate == 0.0 && return x
    istraining() || return x
    if f.mask == nothing
        dims = ntuple(i -> i == dim ? size(x,i) : 1, ndims(x))
        mask = similar(x.data, dims)
        bernoulli!(mask, droprate, scaling)
        f.mask = mask
    end
    x .* Var(f.mask)
end
