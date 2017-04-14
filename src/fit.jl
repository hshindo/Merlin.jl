export fit
import ProgressMeter: Progress, next!

"""
    fit(data_x, data_y, model, opt, [progress=true])
"""
function fit(data_x::Vector, data_y::Vector, model, opt; progress=true)
    length(data_x) == length(data_y) || throw("Length unmatch.")
    progress && (prog = Progress(length(data_x)))
    loss = 0.0
    dict = ObjectIdDict()
    for i in randperm(length(data_x))
        x, y = data_x[i], data_y[i]
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
    loss /= length(data_x)
    loss
end
