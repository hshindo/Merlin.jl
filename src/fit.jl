export fit
using ProgressMeter

"""
    fit(xs, ys, decode, lossfun, opt, [progress=true])
"""
function fit(xs::Vector, ys::Vector, decode, lossfun, opt; progress=true)
    @assert length(xs) == length(ys)
    progress && (prog = Progress(length(xs)))
    loss = 0.0
    dict = ObjectIdDict()
    for i in randperm(length(xs))
        empty!(dict)
        x, y = xs[i], ys[i]
        z = decode(x)
        l = lossfun(y, z)
        loss += sum(l.data)
        vars = gradient!(l)
        for v in vars
            if isempty(v.args) && !isconst(v)
                haskey(dict, v) && continue
                dict[v] = v
                opt(v.data, v.grad)
            elseif typeof(v.f) <: Functor
                haskey(dict, v.f) && continue
                dict[v.f] = v.f
                update!(v.f, opt)
            end
        end
        progress && next!(prog)
    end
    loss
end
