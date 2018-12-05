export ASGD

"""
    ASGD

Averaged Stochastic Gradient Descent.

# Arguments
* rate: learning rate
"""
mutable struct ASGD
    opt
    on::Bool
    dict::Dict
end

function ASGD(opt, on=false)
    ASGD(opt, on, Dict())
end

function (opt::ASGD)(x::Var)
    opt.opt(x)
    opt.on || return
    ax, t = get!(opt.dict, x) do
        a = similar(x.data, size(x))
        fill!(a, 0)
        a, 0
    end
    α = t / (t+1)
    β = 1 / (t+1)
    addto!(α, ax, β, x.data)
    opt.dict[x] = (ax, t+1)
end

function Base.replace!(f, opt::ASGD, params)
    opt.on || return
    pdata = map(p -> p.data, params)
    for p in params
        p.data = opt.dict[p][1]
    end
    y = f()
    for i = 1:length(params)
        params[i].data = pdata[i]
    end
    y
end
