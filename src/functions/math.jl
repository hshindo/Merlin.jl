import Base: broadcast, transpose
import Base: +, -, *, /, ^

doc"""
    exp.(x)
"""
broadcast(::typeof(exp), x::Var) = Var(exp.(x.data), x.batchdims, broadcast, (exp,x))

broadcast(::typeof(exp), x::Node; name="exp") = Node(broadcast, exp, x, name=name)

function addgrad!(y::Var, ::typeof(broadcast), ::typeof(exp), x::Var)
    isvoid(x.grad) || ∇exp!(y.data, y.grad, x.grad)
end

function ∇exp!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
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
    log.(x)
"""
broadcast(::typeof(log), x::Var) = Var(log.(x.data), x.batchdims, broadcast, (log,x))

broadcast(::typeof(log), x::Node; name="log") = Node(broadcast, log, x, name=name)

function addgrad!(y::Var, ::typeof(broadcast), ::typeof(log), x::Var)
    isvoid(x.grad) || ∇log!(y.grad,x.data,x.grad)
end

function ∇log!{T}(gy::Array{T}, x::Array{T}, gx::Array{T})
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
    transpose(x)
"""
transpose(x::Var) = Var(transpose(x.data), x.batchdims, transpose, (x,))

transpose(x::Node; name="transpose") = Node(transpose, x, name=name)

function addgrad!(y::Var, ::typeof(transpose), x::Var)
    isvoid(x.grad) || BLAS.axpy!(eltype(x.data)(1), transpose(y.grad), x.grad)
end

"""
    +(x1::Var, x2::Var)
"""
function +(x1::Var, x2::Var)
    x1.batchdims == x2.batchdims || throw("Batchdims mismatch.")
    Var(x1.data + x2.data, x1.batchdims, +, (x1,x2))
end
+(x1::Union{Number,Array}, x2::Var) = Var(x1) + x2
+(x1::Var, x2::Union{Number,Array}) = x1 + Var(x2)

+(x1::Node, x2; name="+") = Node(+, x1, x2, name=name)
+(x1, x2::Node; name="+") = Node(+, x1, x2, name=name)
+(x1::Node, x2::Node; name="+") = Node(+, x1, x2, name=name)

function addgrad!(y::Var, ::typeof(+), x1::Var, x2::Var)
    T = eltype(y.grad)
    isvoid(x1.grad) || BLAS.axpy!(T(1), y.grad, x1.grad)
    isvoid(x2.grad) || BLAS.axpy!(T(1), y.grad, x2.grad)
end

"""
    -(x1, x2)
"""
function -(x1::Var, x2::Var)
    if isa(x1.data,Array) && isa(x2.data,Array)
        x1.batchdims == x2.batchdims || throw("Batchdims mismatch.")
    end
    Var(x1.data - x2.data, x2.batchdims, -, (x1,x2))
end
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)
-(x::Var) = Var(-x.data, x.batchdims, -, (x,))

-(a::Number, x::Node) = Node(-, Var(a), x)
-(x1::Node, x2::Node) = Node(-, x1, x2)
-(x::Node) = Node(-, x)

function addgrad!(y::Var, ::typeof(-), x1::Var, x2::Var)
    T = eltype(y.grad)
    isvoid(x1.grad) || BLAS.axpy!(T(1), y.grad, x1.grad)
    isvoid(x2.grad) || BLAS.axpy!(T(-1), y.grad, x2.grad)
end

function addgrad!(y::Var, ::typeof(-), x::Var)
    T = eltype(y.grad)
    isvoid(x.grad) || BLAS.axpy!(T(-1), y.grad, x.grad)
end

"""
    .+(x1::Var, x2::Var)
"""
function broadcast(::typeof(+), x1::Var, x2::Var)
    throw("Not implemented yet.")
    Var(x1.data .+ x2.data, broadcast, (+,x1,x2))
end

broadcast(::typeof(+), x1::Node, x2::Node; name=".+") = Node(broadcast, +, x1, x2, name=name)

function addgrad!(y::Var, ::typeof(broadcast), ::typeof(+), x1::Var, x2::Var)
    isvoid(x1.grad) || ∇elemplus!(y.grad, x1.grad)
    isvoid(x2.grad) || ∇elemplus!(y.grad, x2.grad)
end

function ∇elemplus!{T,N}(gy::Array{T,N}, gx::Array{T,N})
    for i = 1:N
        size(gx,i) == 1 && size(gy,i) > 1 && (gy = sum(gy,i))
    end
    BLAS.axpy!(T(1), gy, gx)
end
#∇elemplus!{T,N}(gy::Array{T,N}, gx::Array{T}) = ∇elemplus!(gy, redim(gx,N))

"""
    .-(x1::Var, x2::Var)
"""
function broadcast(::typeof(-), x1::Var, x2::Var)
    throw("Not implemented yet.")
    Var(x1.data .- x2.data, broadcast, (-,x1,x2))
end

broadcast(::typeof(-), x1::Node, x2::Node; name=".-") = Node(broadcast, -, x1, x2, name=name)

function addgrad!(y::Var, ::typeof(broadcast), ::typeof(-), x1::Var, x2::Var)
    isvoid(x1.grad) || ∇elemplus!(y.grad, x1.grad)
    isvoid(x2.grad) || ∇elemminus!(y.grad, x2.grad)
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
function broadcast(::typeof(*), x1::Var, x2::Var)
    x1.batchdims == x2.batchdims || throw("")
    Var(x1.data .* x2.data, x1.batchdims, broadcast, (*,x1,x2))
end

broadcast(::typeof(*), x1::Node, x2::Node; name=".*") = Node(broadcast, *, x1, x2, name=name)

function addgrad!(y::Var, ::typeof(broadcast), ::typeof(*), x1::Var, x2::Var)
    isvoid(x1.grad) || ∇elemtimes!(y.grad, x2.data, x1.grad)
    isvoid(x2.grad) || ∇elemtimes!(y.grad, x1.data, x2.grad)
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
    length(A.batchdims) == 1 || throw("")
    Var(A.data * B.data, B.batchdims, *, (A,B))
end

*(A::Node, B::Node; name="*") = Node(*, A, B, name=name)

function addgrad!(C::Var, ::typeof(*), A::Var, B::Var)
    T = eltype(C.data)
    isvoid(A.grad) || BLAS.gemm!('N', 'T', T(1), C.grad, B.data, T(1), A.grad)
    isvoid(B.grad) || BLAS.gemm!('T', 'N', T(1), A.data, C.grad, T(1), B.grad)
end

"""
    /(x1::Var, a)
"""
/(x::Var, a::Number) = Var(x.data, x.batchdims, /, (x,a))

/(x::Node, a::Number; name="/") = Node(/, x, a, name=name)

function addgrad!(y::Var, ::typeof(/), x::Var, a::Number)
    T = eltype(x.data)
    isvoid(x.grad) || ∇divide!(y.grad, x.grad, T(a))
end

function ∇divide!{T}(gy::Array{T}, gx::Array{T}, a::T)
    @inbounds for i = 1:length(gy)
        gx[i] += gy[i] / a
    end
end

"""
    ^(x::Var, a::Number)
"""
^(x::Var, a::Number) = Var(x.data ^ a, ^, (x,a))
^(x::Node, a::Number) = Node(^, x, a)

function addgrad!(y::Var, ::typeof(^), x::Var, a::Number)
    T = eltype(x.data)
    isvoid(x.grad) || ∇elempow!(T(a), x.data, x.grad, y.data, y.grad)
end

function ∇elempow!{T}(a::T, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
   @inbounds for i = 1:length(gx)
       gx[i] += gy[i] * a * y[i] / x[i]
   end
end
