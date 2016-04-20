"""
BLAS gemm function as follows:
\$ C = \alpha A B + \beta C \$
where A, B amd C are matricies
This is in-place operation.
"""
type GEMM! <: Functor
  alpha::Float64
  beta::Float64
end

@compat (f::GEMM!)(args) = forward(f, args)
function forward!(f::GEMM!, v::Variable)
  T = eltype(v)
  gemm!('N', 'N', T(f.alpha), v[1].value, v[2].value, T(f.beta), v[3].value)
  v.value = v[3].value
  v.backward! = () -> begin
  end
end

function match(pat::Variable, var::Variable)
  typeof(pat.f) == typeof(var.f) || return false
  length(pat.args) <= length(var.args) || return false
  i, j = 1, 1
  while i < length(pat.args) && j < length(var.args)

  end
end

function compile!(::Type{GEMM!}, var::Variable)
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
BLAS axpy function as follows:
\$ y = \alpha x + y \$
This is in-place operation.
"""
type AXPY! <: Functor
  alpha::Float64
end

function forward!(f::AXPY!, v::Variable)
  @assert (length(v.args) == 2)
  T = eltype(v[1])
  axpy!(v[1].value, v[2].value)
end

function compile!(::Type{AXPY!}, var::Variable)
  false
end
