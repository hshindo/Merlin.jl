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
