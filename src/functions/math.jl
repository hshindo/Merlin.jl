import Base: exp, log, broadcast, transpose
import Base: +, -, *, /, ^

doc"""
    exp(x)
"""
exp(x::Var) = Var(exp(x.data), (exp,x))
exp(x::Node; name="") = Node(exp, (x,), name)
exp(x::Array) = exp.(x)

function addgrad!(y::Var, ::typeof(exp), x::Var)
    isvoid(x.grad) || ∇exp!(y.data, y.grad, x.grad)
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

doc"""
    log(x)
"""
log(x::Var) = Var(log(x.data), (log,x))
log(x::Node; name="") = Node(log, (x,), name)
log(x::Array) = log.(x)

function addgrad!(y::Var, ::typeof(log), x::Var)
    isvoid(x.grad) || ∇log!(y.grad, x.data, x.grad)
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

doc"""
    transpose(x)
"""
function transpose(x::Var)
    throw("Not implemented yet.")
    ys = Var[]
    cumdim = 0
    m = size(x, 1)
    for s in x.batchdims
        p = pointer(x.data, m*cumdim+1)
        a = unsafe_wrap(Array, p, (m,s))
        y = Var(transpose(a), [m], transpose, (x,cumdim,cumdim+s-1))
        push!(ys, y)
        cumdim += s
    end
    ys
end
transpose(x::Node; name="") = Node(transpose, (x,), name)

function addgrad!(y::Var, ::typeof(transpose), x::Var, s::Int, e::Int)
    if !isvoid(x.grad)
        x.grad[:,s:e] += transpose(y.grad)
    end
end

doc"""
    +(x1::Var, x2::Var)
    +(a::Number, x::Var)
    +(x::Var, a::Number)
"""
+(x1::Var, x2::Var) = Var(x1.data + x2.data, (+,x1,x2))
+(x1::Union{Number,Array}, x2::Var) = Var(x1) + x2
+(x1::Var, x2::Union{Number,Array}) = x1 + Var(x2)
+(x1::Node, x2::Node; name="") = Node(+, (x1,x2), name)

function addgrad!(y::Var, ::typeof(+), x1::Var, x2::Var)
    T = eltype(y)
    isvoid(x1.grad) || BLAS.axpy!(T(1), y.grad, x1.grad)
    isvoid(x2.grad) || BLAS.axpy!(T(1), y.grad, x2.grad)
end

doc"""
    -(x1, x2)
"""
-(x1::Var, x2::Var) = Var(x1.data - x2.data, (-,x1,x2))
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)

function addgrad!(y::Var, ::typeof(-), x1::Var, x2::Var)
    T = eltype(y)
    isvoid(x1.grad) || BLAS.axpy!(T(1), y.grad, x1.grad)
    isvoid(x2.grad) || BLAS.axpy!(T(-1), y.grad, x2.grad)
end

function -(x::Var)
    Var(-x.data, (-,x))
end
-(x1::Node, x2::Node; name="") = Node(-, (x1,x2), name)
function -(x::Node; name="")
    Node(-, (x,), name)
end

function addgrad!(y::Var, ::typeof(-), x::Var)
    T = eltype(y)
    isvoid(x.grad) || BLAS.axpy!(T(-1), y.grad, x.grad)
end

doc"""
    .+(x1::Var, x2::Var)
"""
broadcast(::typeof(+), x1::Var, x2::Var) = Var(x1.data .+ x2.data, (broadcast,+,x1,x2))
broadcast(::typeof(+), x1::Node, x2::Node; name="") = Node(broadcast, (+,x1,x2), name)

function addgrad!(y::Var, ::typeof(broadcast), ::typeof(+), x1::Var, x2::Var)
    isvoid(x1.grad) || ∇broadcast_plus!(y.grad, x1.grad)
    isvoid(x2.grad) || ∇broadcast_plus!(y.grad, x2.grad)
end

function ∇broadcast_plus!(gy::Array{T}, gx::Array{T}) where T
    dims = Int[]
    for i = 1:ndims(gy)
        size(gx,i) == 1 && size(gy,i) > 1 && push!(dims,i)
    end
    BLAS.axpy!(T(1), sum(gy,dims), gx)
end

doc"""
    .-(x1::Var, x2::Var)
"""
broadcast(::typeof(-), x1::Var, x2::Var) = Var(x1.data .- x2.data, (broadcast,-,x1,x2))
broadcast(::typeof(-), x1::Node, x2::Node; name="") = Node(broadcast, (-,x1,x2), name)

function addgrad!(y::Var, ::typeof(broadcast), ::typeof(-), x1::Var, x2::Var)
    isvoid(x1.grad) || ∇broadcast_plus!(y.grad, x1.grad)
    isvoid(x2.grad) || ∇broadcast_minus!(y.grad, x2.grad)
end

function ∇broadcast_minus!(gy::Array{T}, gx::Array{T}) where T
    dims = Int[]
    for i = 1:ndims(gy)
        size(gx,i) == 1 && size(gy,i) > 1 && push!(dims,i)
    end
    BLAS.axpy!(T(-1), sum(gy,dims), gx)
end

doc"""
    .*(x1::Var, x2::Var)
"""
broadcast(::typeof(*), x1::Var, x2::Var) = Var(x1.data .* x2.data, (broadcast,*,x1,x2))
broadcast(::typeof(*), x1::Node, x2::Node; name="") = Node(broadcast, (*,x1,x2), name)

function addgrad!(y::Var, ::typeof(broadcast), ::typeof(*), x1::Var, x2::Var)
    isvoid(x1.grad) || ∇broadcast_times!(y.grad, x2.data, x1.grad)
    isvoid(x2.grad) || ∇broadcast_times!(y.grad, x1.data, x2.grad)
end

function ∇broadcast_times!(gy::Array{T}, x2::Array{T}, gx1::Array{T}) where T
    g = gy .* x2
    dims = Int[]
    for i = 1:ndims(gy)
        size(gx1,i) == 1 && size(g,i) > 1 && push!(dims,i)
    end
    BLAS.axpy!(T(1), sum(g,dims), gx1)
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

doc"""
    *(A::Var, B::Var)
"""
*(A::Var, B::Var) = Var(A.data * B.data, (*,A,B))
*(A::Node, B::Node; name="") = Node(*, (A,B), name)

function addgrad!(C::Var, ::typeof(*), A::Var, B::Var)
    T = eltype(C)
    isvoid(A.grad) || BLAS.gemm!('N', 'T', T(1), C.grad, B.data, T(1), A.grad)
    isvoid(B.grad) || BLAS.gemm!('T', 'N', T(1), A.data, C.grad, T(1), B.grad)
end

doc"""
    /(x1::Var, a)
"""
function /(x::Var, a::Number)
    y = Var(x.data/a, (x,))
    y.∇! = () -> begin
        T = eltype(x)
        isvoid(x.grad) || ∇divide!(y.grad, x.grad, T(a))
    end
    y
end
/(x::Node, a::Number; name="") = Node(/, (x,a), name)

function ∇divide!(gy::Array{T}, gx::Array{T}, a::T) where T
    @inbounds for i = 1:length(gy)
        gx[i] += gy[i] / a
    end
end

doc"""
    ^(x::Var, a::Number)
"""
^(x::Var, a::Number) = Var(x.data^a, (^,x,a))
^(x::Node, a::Number; name="") = Node(^, (x,a), name)

function addgrad!(y::Var, ::typeof(^), x::Var, a::Number)
    T = eltype(x)
    isvoid(x.grad) || ∇elempow!(T(a), x.data, x.grad, y.data, y.grad)
end

function ∇elempow!(a::T, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T}) where T
   @inbounds for i = 1:length(gx)
       gx[i] += gy[i] * a * y[i] / x[i]
   end
end
