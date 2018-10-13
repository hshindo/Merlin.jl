import Base: +, -, *

"""
    +(x1::Var, x2::Var)
"""
function +(x1::Var, x2::Var)
    Var(x1.data + x2.data, ∇plus!, (x1,x2))
end
+(x1::Node, x2) = Node(+, (x1,x2))
+(x1, x2::Node) = Node(+, (x1,x2))

function ∇plus!(y::Var, x1::Var, x2::Var)
    T = eltype(y)
    isnothing(x1.grad) || addto!(x1.grad,y.grad)
    isnothing(x2.grad) || addto!(x2.grad,y.grad)
end

"""
    -(x1, x2)
"""
function -(x1::Var, x2::Var)
    Var(x1.data - x2.data, ∇minus!, (x1,x2))
end
-(x1::Node, x2) = Node(-, (x1,x2))
-(x1, x2::Node) = Node(-, (x1,x2))

function ∇minus!(y::Var, x1::Var, x2::Var)
    T = eltype(y)
    isnothing(x1.grad) || addto!(x1.grad, y.grad)
    isnothing(x2.grad) || axpy!(T(-1), y.grad, x2.grad)
end

"""
    *(A::Var, B::Var)
"""
function *(A::Var, B::Var)
    Var(A.data * B.data, ∇times!, (A,B))
end
*(x1::Node, x2) = Node(*, (x1,x2))
*(x1, x2::Node) = Node(*, (x1,x2))

function ∇times!(C::Var, A::Var, B::Var)
    T = eltype(C)
    isnothing(A.grad) || gemm!('N', 'T', T(1), C.grad, B.data, T(1), A.grad)
    isnothing(B.grad) || gemm!('T', 'N', T(1), A.data, C.grad, T(1), B.grad)
end

#=
"""
    exp(x)
"""
function exp(x::Var)
    configure!(x)
    Var(exp(x.data), (exp,x))
end
exp(x::Array) = exp.(x)

function addgrad!(y::Var, ::typeof(exp), x::Var)
    isvoid(x.grad) && return
    ∇exp!(y.data, y.grad, x.grad)
end

function ∇exp!(y::Array{T}, gy::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i]
    end
end

"""
    log(x)
"""
function log(x::Var)
    configure!(x)
    Var(log(x.data), (log,x))
end
log(x::Array) = log.(x)

function addgrad!(y::Var, ::typeof(log), x::Var)
    isvoid(x.grad) && return
    ∇log!(y.grad, x.data, x.grad)
end

function ∇log!(gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] / x[i]
    end
end

@generated function ∇exp!(y::CuArray{T}, gy::CuArray{T}, gx::CuArray{T}) where T
    f = CuFunction("""
    __global__ void f($T *y, $T *gy, $T *gx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            gx[idx] += gy[idx] * y[idx];
        }
    }""")
    quote
        $f(y.ptr, gy.ptr, gx.ptr, length(y), dx=length(y))
    end
end

@generated function ∇log!(gy::CuArray{T}, x::CuArray{T}, gx::CuArray{T}) where T
    f = CuFunction("""
    __global__ void f($T *gy, $T *x, $T *gx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            gx[idx] += gy[idx] / x[idx];
        }
    }""")
    quote
        $f(gy.ptr, x.ptr, gx.ptr, length(gy), dx=length(gy))
    end
end

function ∇elemtimes!{T}(gy::Array{T}, x2::Array{T}, gx1::Array{T})
    ind_x2 = CartesianIndex(size(x2))
    ind_gx1 = CartesianIndex(size(gx1))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx1[min(ind_gx1,I)] += gy[I] * x2[min(ind_x2,I)]
    end
end
=#
