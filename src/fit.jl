export fit
using ProgressMeter

"""
    fit(xs, ys, decode, lossfun, opt, [progress=true])
"""
function fit(xs::Vector, ys::Vector, decode, lossfun, opt; progress=true)
    @assert length(xs) == length(ys)
    progress && (prog = Progress(length(xs)))
    loss = 0.0
    fdict = ObjectIdDict()
    for i in randperm(length(xs))
        empty!(fdict)
        x, y = xs[i], ys[i]
        z = decode(x)
        l = lossfun(y, z)
        loss += sum(l.data)
        vars = gradient!(l)
        for v in vars
            isempty(v.args) || continue
            isconst(v) && continue
            haskey(fdict, v) && continue
            fdict[v] = v
            opt(v.data, v.grad)
        end
        progress && next!(prog)
    end

    loss
end
