export fit
using ProgressMeter

function fit(decode, lossfun, opt, xs::Vector, ys::Vector)
    @assert length(xs) == length(ys)
    prog = Progress(length(xs))
    loss = 0.0
    for i in randperm(length(xs))
        z = decode(xs[i])
        out = lossfun(ys[i], z)
        loss += sum(out.data)
        vars = gradient!(out)
        for v in vars
            typeof(v.f) <: Functor && update!(v.f, opt)
        end
        #next!(prog)
    end
    loss
end
