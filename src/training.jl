export fit

function fit(xs, ys, nn, lossfun, opt::Optimizer)
  loss = 0.0
  for i in randperm(length(xs))
    z = nn(xs[i])
    l = lossfun(ys[i], z)
    loss += sum(l.value)
    vars = backward!(l)
    for v in vars
      length(v.args) == 0 && hasgrad(v) && update!(opt, v.value, v.grad)
    end
  end
  loss
end
