export checkgrad, checkcuda

# TODO: check functor
function checkgrad(f, args...; eps=1e-3)
    vars = collect(filter(a -> isa(a,Var) && !isvoid(a.grad), args))
    foreach(zerograd!, vars)
    y = f(args...)
    gradient!(y)
    gxs1 = map(x -> x.grad, vars)
    gxs2 = map(vars) do v
        x = v.data
        gx = similar(x)
        for k = 1:length(x)
            xk = x[k]
            x[k] = xk + eps
            y1 = copy(f(args...).data)
            x[k] = xk - eps
            y2 = copy(f(args...).data)
            x[k] = xk
            gx[k] = sum(y1 - y2) / 2eps
        end
        gx
    end
    foreach(zip(gxs1,gxs2)) do g
        diff = g[1] - g[2]
        if any(d -> abs(d) >= eps, diff)
            println(maximum(d -> abs(d), diff))
            println(diff)
            #println(g[2])
            throw("")
        end
    end
    usecuda && checkcuda(f, args..., eps=eps)
    true
end

function checkcuda(f, args...; eps=1e-3)
    vars = collect(filter(a -> isa(a,Var) && !isvoid(a.grad), args))
    foreach(zerograd!, vars)
    y1 = f(args...)
    gradient!(y1)
    gxs1 = map(x -> x.grad, vars)
    y1 = y1.data

    foreach(v -> setbackend!(v,CuArray), vars)
    foreach(zerograd!, vars)

    y2 = f(args...)
    gradient!(y2)
    gxs2 = map(x -> Array(x.grad), vars)
    y2 = Array(y2.data)

    if !all(d -> abs(d) < eps, y1 - y2)
        throw("Output of CPU and CUDA mismatch.")
    end
    foreach(zip(gxs1,gxs2)) do g
        diff = g[1] - g[2]
        if !all(d -> abs(d) < eps, diff)
            println(g[1])
            println(g[2])
            throw(diff)
        end
    end

    foreach(v -> setbackend!(v,Array), vars)
    foreach(zerograd!, vars)
    true
end
