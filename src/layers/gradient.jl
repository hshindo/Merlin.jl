export topsort, gradient!, approx_grad, @checkgrad

function topsort(top)
  sorted = []
  dict = ObjectIdDict()
  function visit(node)
    haskey(dict, node) && return
    dict[node] = node
    for t in tails(node)
      visit(t)
    end
    push!(sorted, node)
  end
  visit(top)
  sorted
end

hasgrad(l::Layer) = l.gy != nothing

isleaf(l::Layer) = isempty(tails(l))

function gradient!(top::Layer)
  sorted = topsort(top)
  hasgrad(top) || (top.gy = ones(top.y))
  for i = 1:length(sorted)-1 # excludes top
    l = sorted[i]
    isleaf(l) && continue
    l.gy = zeros(l.y)
  end
  for i = length(sorted):-1:1
    l = sorted[i]
    backward!(l)
  end
  sorted
end

const gradeps = 1e-3

"""
Compute numerical gradient.
"""
function approx_grad{T<:Layer}(f, args::Vector{T})
  map(args) do v
    x = v.y
    gx = similar(x)
    for k = 1:length(x)
      xk = x[k]
      x[k] = xk + gradeps
      y1 = f().y
      x[k] = xk - gradeps
      y2 = f().y
      x[k] = xk
      gx[k] = sum(y1 - y2) / (2gradeps)
    end
    gx
  end
end

macro checkgrad(f, args)
  quote
    local f() = $(esc(f))
    local args = $(esc(args))
    for x in args
      x.gy = zeros(x.y)
    end
    y = f()
    gradient!(y)
    approx_gxs = approx_grad(f, args)
    for i = 1:length(args)
      gx1 = args[i].gy
      gx2 = approx_gxs[i]
      all(d -> abs(d) < 1e-3, gx1 - gx2) && continue
      println(gx1 - gx2)
      return false
    end
    true
  end
end
