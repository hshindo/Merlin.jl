export topsort, gradient!, approx_grad, @checkgrad

function gradient!(top::Var)
    sorted = topsort(top)
    hasgrad(top) || (top.grad = ones(top.data))
    for i = 1:length(sorted)-1 # excludes top
        v = sorted[i]
        isempty(v.tails) && continue
        v.grad = zeros(v.data)
    end
    for i = length(sorted):-1:1
        backward!(sorted[i])
    end
    sorted
end

const gradeps = 1e-2

"""
Compute numerical gradient.
"""
function approx_grad{T}(f, args::Vector{T})
    map(args) do v
        x = v.data
        gx = similar(x)
        for k = 1:length(x)
            xk = x[k]
            x[k] = xk + gradeps
            y1 = f().data
            x[k] = xk - gradeps
            y2 = f().data
            x[k] = xk
            gx[k] = sum(y1 - y2) / (2gradeps)
        end
        gx
    end
end

macro checkgrad(f, args)
    quote
        local f() = $(esc(f))
        local args = $(esc(args))
        for x in args
            x.grad = zeros(x.data)
        end
        y = f()
        gradient!(y)
        approx_gxs = approx_grad(f, args)
        for i = 1:length(args)
            gx1 = args[i].grad
            gx2 = approx_gxs[i]
            all(d -> abs(d) < gradeps, gx1 - gx2) && continue
            println(gx1 - gx2)
            return false
        end
        true
    end
end
