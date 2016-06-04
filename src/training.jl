export fit

function fit(decode, lossfun, opt, xs, ys)
  loss = 0.0
  for i in randperm(length(xs))
    z = decode(xs[i])
    l = lossfun(ys[i], z)
    loss += sum(l.value)
    vars = gradient!(l)
    for v in vars
      isempty(v.args) && hasgrad(v) && update!(opt, v.value, v.grad)
    end
  end
  loss
end
