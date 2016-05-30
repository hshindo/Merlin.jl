"""
Compute numerical gradient.
"""
function approx_grad(f, args::Tuple, eps)
  map(args) do x
    x = x.value
    gx = similar(x)
    for k = 1:length(x)
      xk = x[k]
      x[k] = xk + eps
      y1 = f(args...).value
      x[k] = xk - eps
      y2 = f(args...).value
      x[k] = xk
      gx[k] = sum(y1 - y2) / 2eps
    end
    gx
  end
end

"""
Check gradient.
"""
function checkgrad(f, args::Tuple)
  const eps = 1e-3
  y = f(args...)
  for x in args
    x.grad = zeros(x.value)
  end
  gradient!(y)
  approx_gxs = approx_grad(f, args, eps)
  for i = 1:length(args)
    gx1 = args[i].grad
    gx2 = approx_gxs[i]
    if any(d -> abs(d) >= 2eps, gx1 - gx2)
      println(gx1 - gx2)
      return false
    end
  end
  true
end
checkgrad(f, args::Var...) = checkgrad(f, args)
