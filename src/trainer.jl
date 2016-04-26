export fit

function fit{T,X,Y}(nn::T, xs::Vector{X}, ys::Vector{Y})
  loss = 0.0
  for i in randperm(length(xs))
    z = nn(xs[i])
    l = nn.lossfun(ys[i], z)
    loss += sum(l.value)
    vars = backward!(l)
    for v in vars
      applicable(update!, nn.opt, v.f) && update!(nn.opt, v.f)
      length(v.args) == 0 && hasgrad(v) && update!(nn.opt, v.value, v.grad)
    end
  end
  loss
end
