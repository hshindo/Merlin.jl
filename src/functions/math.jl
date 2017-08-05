import Base: exp, log, transpose
import Base: +, -, *, /, pow

"""
    exp(x::Var)
"""
function exp(x::Var)
    y = Var(exp.(x.data), exp, (x,))
    y.df! = () -> begin
        isvoid(x.grad) && return
        ∇exp!(y.data, y.grad, x.grad)
    end
    y
end

function ∇exp!(y::Array{T}, gy::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i]
    end
end

#=
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
=#

"""
    log(x::Var)
"""
function log(x::Var)
    y = Var(nothing, log, (x,))
    y.data = log.(x.data)
    y.df! = () -> begin
        isvoid(x.grad) && return
        ∇log!(y.grad, x.data, x.grad)
    end
    y
end

function ∇log!(gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] / x[i]
    end
end

#=
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
=#

"""
    transpose(x::Var)
"""
function transpose(x::Var)
    isvoid(x.batchdims) || throw("")
    y = Var(transpose(x.data), x.batchdims, transpose, (x,))
    y.df! = function df!()
        isvoid(x.grad) || BLAS.axpy!(eltype(x.data)(1), transpose(y.grad), x.grad)
    end
    y
end

"""
    +(x1::Var, x2::Var)
"""
function +(x1::Var, x2::Var)
    x1.batchdims == x2.batchdims || throw("batchdims unmatch")
    data = x1.data + x2.data
    Var(data, x1.batchdims, +, (x1,x2))
end
+(x1::Union{Number,Array}, x2::Var) = Var(x1) + x2
+(x1::Var, x2::Union{Number,Array}) = x1 + Var(x2)
+(x1::Node, x2) = Node(+, x1, x2)
+(x1, x2::Node) = Node(+, x1, x2)
+(x1::Node, x2::Node) = Node(+, x1, x2)

function addgrad!(y::Var, ::typeof(+), x1::Var, x2::Var)
    T = eltype(y.grad)
    isvoid(x1.grad) || BLAS.axpy!(T(1), y.grad, x1.grad)
    isvoid(x2.grad) || BLAS.axpy!(T(1), y.grad, x2.grad)
end

"""
    .+(x1::Var, x2::Var)
"""
function Base.broadcast(::typeof(+), x1::Var, x2::Var)
    data = broadcast(+, x1.data, x2.data)
    batchdims = nothing
    Var(data, batchdims, broadcast, (+,x1,x2))
end

function gradient!(y::Var, ::typeof(broadcast), ::typeof(+), x1::Var, x2::Var)
    isvoid(x1.grad) || ∇elemplus!(y.grad, x1.grad)
    isvoid(x2.grad) || ∇elemplus!(y.grad, x2.grad)
end

function ∇elemplus!(gy::Array{T,N}, gx::Array{T,N}) where {T,N}
    for i = 1:N
        size(gx,i) == 1 && size(gy,i) > 1 && (gy = sum(gy,i))
    end
    BLAS.axpy!(T(1), gy, gx)
end
#∇elemplus!{T,N}(gy::Array{T,N}, gx::Array{T}) = ∇elemplus!(gy, redim(gx,N))

"""
    -(x1::Var, x2::Var)
    -(x::Var)
"""
function -(x1::Var, x2::Var)
    y = Var(nothing, -, (x1,x2))
    y.data = x1.data - x2.data
    y.df! = () -> begin
        T = eltype(y.grad)
        isvoid(x1.grad) || BLAS.axpy!(T(1), y.grad, x1.grad)
        isvoid(x2.grad) || BLAS.axpy!(T(-1), y.grad, x2.grad)
    end
    y
end

function -(x::Var)
    y = Var(nothing, -, (x,))
    y.data = -x.data
    y.df! = () -> begin
        isvoid(x.grad) && return
        T = eltype(y.grad)
        BLAS.axpy!(T(-1), y.grad, x.grad)
    end
    y
end
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)

"""
    .-(x1::Var, x2::Var)
"""
function Base.broadcast(x1::Var, x2::Var)
    y = Var(nothing, broadcast, (-,x1,x2))
    y.data = broadcast(-, x1.data, x2.data)
    y.df! = () -> begin
        isvoid(x1.grad) || ∇elemplus!(y.grad, x1.grad)
        isvoid(x2.grad) || ∇elemminus!(y.grad, x2.grad)
    end
    y
end

function ∇elemminus!{T,N}(gy::Array{T,N}, gx::Array{T,N})
    for i = 1:N
        size(gx,i) == 1 && size(gy,i) > 1 && (gy = sum(gy,i))
    end
    BLAS.axpy!(T(-1), gy, gx)
end

#=
function ∇elemminus!{T}(gy::Array{T}, gx::Array{T})
    ind_gx = CartesianIndex(size(gx))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx[min(ind_gx,I)] -= gy[I]
    end
end
=#

"""
    \.\*(x1::Var, x2::Var)
"""
function Base.broadcast(::typeof(*), x1::Var, x2::Var)
    y = Var(nothing, broadcast, (*,x1,x2))
    y.data = broadcast(*, x1.data, x2.data)
    y.df! = () -> begin
        isvoid(x1.grad) || ∇elemtimes!(y.grad, x2.data, x1.grad)
        isvoid(x2.grad) || ∇elemtimes!(y.grad, x1.data, x2.grad)
    end
    y
end

function ∇elemtimes!{T,N}(gy::Array{T,N}, x2::Array{T,N}, gx1::Array{T,N})
    if size(x2) == size(gx1)
        @inbounds for i = 1:length(gy)
            gx1[i] += gy[i] * x2[i]
        end
    else
        gx = gy .* x2
        for i = 1:N
            size(gx1,i) == 1 && size(gx,i) > 1 && (gx = sum(gx,i))
        end
        BLAS.axpy!(T(1), gx, gx1)
    end
end

#=
@generated function ∇elemtimes!{T,N}(gy::CuArray{T,N}, x2::CuArray{T,N}, gx1::CuArray{T,N})
    f = CuFunction("""
    __global__ void f(Array<$T,$N> gy, Array<$T,$N> x2, Array<$T,$N> gx1) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < gy.length()) {
            gx1[idx] += gy[idx] * x2[idx];
        }
    }""")
    quote
        if size(x2) == size(gx1)
            $f(gy, x2, gx1, dx=length(gy))
        else
            gx = gy .* x2
            for i = 1:N
                size(gx1,i) == 1 && size(gx,i) > 1 && (gx = sum(gx,i))
            end
            BLAS.axpy!(T(1), gx, gx1)
        end
    end
end
=#

#=
function ∇elemtimes!{T}(gy::Array{T}, x2::Array{T}, gx1::Array{T})
    ind_x2 = CartesianIndex(size(x2))
    ind_gx1 = CartesianIndex(size(gx1))
    @inbounds @simd for I in CartesianRange(size(gy))
        gx1[min(ind_gx1,I)] += gy[I] * x2[min(ind_x2,I)]
    end
end
=#

"""
    \*(A::Var, B::Var)
"""
function *(A::Var, B::Var)
    data = A.data * B.data
    length(A.batchdims) == 1 || throw("Not implemented yet.")
    batchdims = B.batchdims
    Var(data, batchdims, *, (A,B))
end

function gradient!(C::Var, f::typeof(*), A::Var, B::Var)
    T = eltype(C.data)
    isvoid(A.grad) || BLAS.gemm!('N', 'T', T(1), C.grad, B.data, T(1), A.grad)
    isvoid(B.grad) || BLAS.gemm!('T', 'N', T(1), A.data, C.grad, T(1), B.grad)
end

"""
    /(x1::Var, a)
"""
function /(x::Var, a::Number)
    a = eltype(x.data)(a)
    y = Var(nothing, /, (x,a))
    y.data = x.data / a
    y.df! = () -> begin
        isvoid(x.grad) && return
        ∇divide!(y.grad, x.grad, a)
    end
    y
end

function ∇divide!{T}(gy::Array{T}, gx::Array{T}, a::T)
    @inbounds for i = 1:length(gy)
        gx[i] += gy[i] / a
    end
end

"""
    .^(x::Var, a::Number)
"""
function .^(x::Var, a::Number)
    y = x.data .^ a
    df(gy) = hasgrad(x) && ∇elemexp!(a, x.data, x.grad, y, gy)
    Var(y, [x], .^, df)
end

function ∇elempow!{T}(a, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
   @inbounds @simd for i = 1:length(gx)
       gx[i] += gy[i] * T(a) * y[i] / x[i]
   end
   gx
end
