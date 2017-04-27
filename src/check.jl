export checkgrad

function checkgrad(f, vars::Var...; eps=1e-3)
    foreach(zerograd!, vars)
    y = f()
    gradient!(y)
    gxs1 = map(x -> x.grad, vars)
    gxs2 = map(vars) do v
        x = v.data
        gx = similar(x)
        for k = 1:length(x)
            xk = x[k]
            x[k] = xk + eps
            y1 = copy(f().data)
            x[k] = xk - eps
            y2 = copy(f().data)
            x[k] = xk
            gx[k] = sum(y1-y2) / 2eps
        end
        gx
    end
    for i = 1:length(vars)
        g1, g2 = gxs1[i], gxs2[i]
        diff = g1 - g2
        if any(d -> abs(d) > eps, diff)
            println(maximum(d -> abs(d), diff))
            println(diff)
            throw("Gradient error.")
        end
    end
    true
end
