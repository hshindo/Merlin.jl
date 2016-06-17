type SumProduct
  coeffs::Vector{Float64}
  ops::Vector{Function}
end

@compat function (f::SumProduct)(xs::Vector{Var})
  for i = 1:2:length(xs)
    xs[i]
  end
end

"""
BLAS gemm function:

$ C = \alpha A B + \beta C $
where A, B amd C are matricies.
"""
type GEMM!
  alpha::Float64
  beta::Float64
end

@compat (f::GEMM!)(x) = forward0(f, x)

function forward!(f::GEMM!, v::Var)
  T = eltype(v)
  gemm!('N', 'N', T(f.alpha), v[1].value, v[2].value, T(f.beta), v[3].value)
  v.value = v[3].value
end

function match(pat::Var, var::Var)
  typeof(pat.f) == typeof(var.f) || return false
  length(pat.args) <= length(var.args) || return false
  i, j = 1, 1
  while i < length(pat.args) && j < length(var.args)

  end
end

function compile!(::Type{GEMM!}, var::Var)
  typeof(var.f) == Add || return false
  index = findfirst(a -> typeof(a.f) == Multiply, var.args)
  index == 0 && return false
  m = var[index]
  pat = Variable()
  args = copy(var[index].args)
  ind3 = ind == 1 ? 2 : 1
  push!(args, var[ind3])
  var.f = GEMM!(1.0,1.0)
  var.args = args
  true
end

"""
BLAS axpy function:
$ y = \alpha x + y $
"""
type AXPY!
  alpha::Float64
end

function forward!(f::AXPY!, v::Var)
  @assert (length(v.args) == 2)
  T = eltype(v[1])
  axpy!(v[1].value, v[2].value)
end

function compile!(::Type{AXPY!}, var::Var)
  false
end
