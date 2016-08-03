export fit
using ProgressMeter

function fit(decode, lossfun, optimize, xs::Vector, ys::Vector)
    @assert length(xs) == length(ys)
    prog = Progress(length(xs))
    loss = 0.0
    for i in randperm(length(xs))
        z = decode(xs[i])
        out = lossfun(ys[i], z)
        loss += sum(out.data)
        vars = gradient!(out)
        for v in vars
            optimize(v.data, v.grad)
        end
        next!(prog)
    end
    loss
end
