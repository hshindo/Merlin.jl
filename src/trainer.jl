export Trainer, fit

type Trainer
  f
  lossfun
  opt::Optimizer
end

function fit(t::Trainer, xs, ys)
  loss = 0.0
  for i in randperm(length(xs))
    z = t.f(xs[i])
    l = t.lossfun(ys[i], z)
    loss += sum(l.val)
    vars = gradient!(l)
    for v in vars
      isempty(v.args) && hasgrad(v) && update!(t.opt, v.val, v.grad)
    end
  end
  loss
end
