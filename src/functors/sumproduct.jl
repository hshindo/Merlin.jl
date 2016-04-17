export SumProduct

"""
## SumProduct

\$ y = p1 * x1 + p2 * x2 + ... \$
where \$p\$ and \$x\$ are matricies.
"""
type SumProduct <: Functor
  weights::Vector{Float64}
end

@compat function (f::SumProduct)(args)
  forward(f, args)
end

function forward!(f::SumProduct, v::Variable)
  for i = 1:length(f.weights)
    vv = v[2i-1], v[2i]

  end
end

function forward(f::SumProduct, xs::Vector{Array})
  params = map(p -> p.value, f.params)
  v.value = sumproduct(params, xs)
end

function sumproduct{T}(params::Vector{Matrix{T}}, xs::Vector{Matrix{T}})
  y = zeros(T, size(params[1],1), size(xs[1],2))
  for i = 1:length(xs)
    gemm!('N', 'N', T(1), params[i], xs[i], T(1), y)
  end
  y
end
