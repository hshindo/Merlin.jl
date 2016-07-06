export Linear

type Linear <: Layer
  w
  b
  x
  y
  gy
end

function Linear(T::Type, indim::Int, outdim::Int)
  x = sqrt(6 / (indim + outdim))
  r = rand(outdim, indim) * 2x - x
  w = Matrix{T}(r)
  b = fill(T(0), outdim, 1)
  Linear(Data(w,zeros(w)), Data(b,zeros(b)), nothing, nothing, nothing)
end

@compat (l::Linear)(x::Layer) = Linear(l.w, l.b, x)
@compat (l::Linear)(x::GraphNode) = GraphNode(l, x)

function Linear(w, b, x)
  y = linear(w.y, b.y, x.y)
  Linear(w, b, x, y, nothing)
end

function linear{T}(w::Matrix{T}, b::Matrix{T}, x::Matrix{T})
  y = w * x
  broadcast!(.+, y, y, b)
  y
end

tails(l::Linear) = [l.w, l.b, l.x]

function backward!(l::Linear)
  T = eltype(l.y)
  hasgrad(l.w) && BLAS.gemm!('N', 'T', T(1), l.gy, l.x.y, T(1), l.w.gy)
  hasgrad(l.x) && BLAS.gemm!('T', 'N', T(1), l.w.y, l.gy, T(1), l.x.gy)
end
