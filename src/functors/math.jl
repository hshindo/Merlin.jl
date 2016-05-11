export Add, ElemAdd
export Subtract, ElemSubtract
export Mult, ElemMult

type Add <: Functor
end
type ElemAdd <: Functor
end
type Subtract <: Functor
end
type ElemSubtract <: Functor
end
type Mult <: Functor
end
type ElemMult <: Functor
end

import Base.+
+(v1::Var, v2::Var) = Add()(v1, v2)
+(x::DataArray, v::Var) = Add()(Var(x), v)
+(v::Var, x::DataArray) = Add()(v, Var(x))

import Base.(.+)
.+(v1::Var, v2::Var) = ElemAdd()(v1, v2)
.+(x, v::Var) = ElemAdd()(Var(x), v)
.+(v::Var, x) = ElemAdd()(v, Var(x))

import Base.-
-(v1::Var, v2::Var) = Subtract()(v1, v2)
-(x, v::Var) = Subtract()(Var(x), v)
-(v::Var, x) = Subtract()(v, Var(x))
-(v::Var) = 0 - v

import Base.(.-)
.-(v1::Var, v2::Var) = ElemSubtract()(v1, v2)
.-(x, v::Var) = ElemSubtract()(Var(x), v)
.-(v::Var, x) = ElemSubtract()(v, Var(x))

import Base.*
*(v1::Var, v2::Var) = Mult()(v1, v2)
*(x, v::Var) = Mult()(Var(x), v)
*(v::Var, x) = Mult()(v, Var(x))

import Base.(.*)
.*(v1::Var, v2::Var) = ElemMult()(v1, v2)
.*(x, v::Var) = ElemMult()(Var(x), v)
.*(v::Var, x) = ElemMult()(v, Var(x))

function forward(f::Add, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val + x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && BLAS.axpy!(T(1), gy, x1.grad)
    hasgrad(x2) && BLAS.axpy!(T(1), gy, x2.grad)
  end
  Var(y, nothing, f, args, backward!)
end

function forward(f::ElemAdd, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val .+ x2.val
  backward! = gy -> begin
    hasgrad(x1) && ∇elemadd!(x1.grad, gy)
    hasgrad(x2) && ∇elemadd!(x2.grad, gy)
  end
  Var(y, nothing, f, args, backward!)
end

function ∇elemadd!{T,N}(gx::Array{T,N}, gy::Array{T,N})
  for offset = 1:length(gx):length(gy)
    BLAS.axpy!(length(gx), T(1), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

function forward(f::Subtract, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val - x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && BLAS.axpy!(T(1), gy, x1.grad)
    hasgrad(x2) && BLAS.axpy!(T(-1), gy, x2.grad)
  end
  Var(y, nothing, f, args, backward!)
end

function forward(f::ElemSubtract, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val .- x2.val
  backward! = gy -> begin
    hasgrad(x1) && ∇elemsubtract!(1.0, x1.grad, gy)
    hasgrad(x2) && ∇elemsubtract!(-1.0, x2.grad, gy)
  end
  Var(y, nothing, f, args, backward!)
end

function ∇elemsubtract!{T,N}(a::Float64, gx::Array{T,N}, gy::Array{T,N})
  for offset = 1:length(gx):length(gy)
    BLAS.axpy!(length(gx), T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

function forward(f::Mult, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val * x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.val, T(1), x1.grad)
    hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.val, gy, T(1), x2.grad)
  end
  Var(y, nothing, f, args, backward!)
end

function forward(f::ElemMult, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val .* x2.val
  backward! = gy -> begin
    hasgrad(x1) && ∇elemmult!(x2.val, x1.grad, gy)
    hasgrad(x2) && ∇elemmult!(x1.val, x2.grad, gy)
  end
  Var(y, nothing, f, args, backward!)
end

function ∇elemmult!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx1)
    gx1[i] += gy[i] * x2[i]
  end
end
