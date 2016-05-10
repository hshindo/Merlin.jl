export Add, ElemAdd
export Subtract, ElemSubtract
export Multiply, ElemMultiply

type Add <: Functor
end
type ElemAdd <: Functor
end
type Subtract <: Functor
end
type ElemSubtract <: Functor
end
type Multiply <: Functor
end
type ElemMultiply <: Functor
end

@compat (f::Add)(xs) = forward(f, xs)
@compat (f::ElemAdd)(xs) = forward(f, xs)
@compat (f::Subtract)(xs) = forward(f, xs)
@compat (f::ElemSubtract)(xs) = forward(f, xs)
@compat (f::Multiply)(xs) = forward(f, xs)
@compat (f::ElemMultiply)(xs) = forward(f, xs)

import Base.+
+(v1::Var, v2::Var) = (v1, v2) |> Add()
+(x, v::Var) = (Var(x), v) |> Add()
+(v::Var, x) = (Var(x), v) |> Add()

import Base.(.+)
.+(v1::Var, v2::Var) = (v1, v2) |> ElemAdd()
.+(x, v::Var) = (Var(x), v) |> ElemAdd()
.+(v::Var, x) = (Var(x), v) |> ElemAdd()

import Base.-
-(v1::Var, v2::Var) = (v1, v2) |> Subtract()
-(x, v::Var) = (Var(x), v) |> Subtract()
-(v::Var, x) = (Var(x), v) |> Subtract()
-(v::Var) = 0 - v

import Base.(.-)
.-(v1::Var, v2::Var) = (v1, v2) |> ElemSubtract()
.-(x, v::Var) = (Var(x), v) |> ElemSubtract()
.-(v::Var, x) = (Var(x), v) |> ElemSubtract()

import Base.*
*(v1::Var, v2::Var) = (v1, v2) |> Multiply()
*(x, v::Var) = (Var(x), v) |> Multiply()
*(v::Var, x) = (Var(x), v) |> Multiply()

import Base.(.*)
.*(v1::Var, v2::Var) = (v1, v2) |> ElemMultiply()
.*(x, v::Var) = (Var(x), v) |> ElemMultiply()
.*(v::Var, x) = (Var(x), v) |> ElemMultiply()

function forward(f::Add, xs::Vector{Var})
  x1, x2 = xs[1], xs[2]
  y = x1.val + x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && BLAS.axpy!(T(1), gy, x1.grad)
    hasgrad(x2) && BLAS.axpy!(T(1), gy, x2.grad)
  end
  Var(y, nothing, f, xs, backward!)
end

function forward(f::ElemAdd, xs::Vector{Var})
  x1, x2 = xs[1], xs[2]
  y = x1.val .+ x2.val
  backward! = gy -> begin
    hasgrad(x1) && backward!(f, x1.grad, gy)
    hasgrad(x2) && backward!(f, x2.grad, gy)
  end
  Var(y, nothing, f, xs, backward!)
end

function backward!{T,N}(f::ElemAdd, gx::Array{T,N}, gy::Array{T,N})
  for offset = 1:length(gx):length(gy)
    BLAS.axpy!(length(gx), T(1), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

function forward(f::Subtract, xs::Vector{Var})
  x1, x2 = xs[1], xs[2]
  y = x1.val - x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && axpy!(T(1), gy, x1.grad)
    hasgrad(x2) && axpy!(T(-1), gy, x2.grad)
  end
  Var(y, nothing, f, xs, backward!)
end


function forward(f::ElemSubtract, xs::Vector{Var})
  x1, x2 = xs[1], xs[2]
  y = x1.val .- x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && backward!(f, 1.0, x1.grad, gy)
    hasgrad(x2) && backward!(f, -1.0, x2.grad, gy)
  end
  Var(y, nothing, f, xs, backward!)
end

function backward!{T,N}(f::ElemSubtract, a::Float64, gx::Array{T,N}, gy::Array{T,N})
  for offset = 1:length(gx):length(gy)
    axpy!(length(gx), T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

function forward(f::Multiply, xs::Vector{Var})
  x1, x2 = xs[1], xs[2]
  y = x1.val * x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.val, T(1), x1.grad)
    hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.val, gy, T(1), x2.grad)
  end
  Var(y, nothing, f, xs, backward!)
end

function forward(f::ElemMultiply, xs::Vector{Var})
  x1, x2 = xs[1], xs[2]
  y = x1.val .* x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && backward!(f, x2.val, x1.grad, gy)
    hasgrad(x2) && backward!(f, x1.val, x2.grad, gy)
  end
end

function backward!{T,N}(f::ElemMultiply, x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx1)
    gx1[i] += gy[i] * x2[i]
  end
end
