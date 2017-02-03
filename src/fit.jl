export fit
using ProgressMeter

"""
    fit(data, model, opt, [progress=true])
"""
function fit(data::Vector, model, opt; progress=true)
    progress && (prog = Progress(length(data)))
    loss = 0.0
    dict = ObjectIdDict()
    for (x,y) in shuffle(data)
        z = model(x, y)
        loss += sum(z.data)
        vars = gradient!(z)
        for v in vars
            isparam(v) && opt(v.data, v.grad)
            isa(v.f, Functor) && (dict[v.f] = v.f)
        end
        foreach(f -> update!(f,opt), keys(dict))
        progress && next!(prog)
    end
    loss /= length(data)
    loss
end

"""
    fit(xs, ys, decode, lossfun, opt, [progress=true])
"""
function fit2(xs::Vector, ys::Vector, decode, lossfun, opt; progress=true)
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
            if isempty(v.args) && !isvoid(v.grad)
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
