export fit
using ProgressMeter

"""
    fit(xs, ys, decode, lossfun, opt, [progress=true])
"""
function fit(xs::Vector, ys::Vector, decode, lossfun, opt; progress=true)
    progress && (prog = Progress(length(xs)))
    loss = 0.0
    for i in randperm(length(xs))
        x, y = xs[i], ys[i]
        z = decode(x)
        l = lossfun(y, z)
        loss += sum(l.data)
        vars = gradient!(l)
        for v in vars
            typeof(v.f) <: Functor && update!(v.f, opt)
        end
        progress && next!(prog)
    end
    loss
end
