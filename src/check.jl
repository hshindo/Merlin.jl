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
        all(d -> abs(d) < eps, diff) || throw((g[1],g[2]))
    end
    #use_cuda && checkcuda(f, args..., eps=eps)
    true
end

function checkcuda(f, args...; eps=1e-3)
    vars = collect(filter(a -> isa(a,Var) && !isvoid(a.grad), args))
    foreach(zerograd!, vars)
    y1 = f(args...)
    gradient!(y1)
    gxs1 = map(x -> x.grad, vars)
    y1 = y1.data

    for v in vars
        v.data = CuArray(v.data)
        v.grad = zeros(v.data)
    end
    y2 = f(args...)
    gradient!(y2)
    gxs2 = map(x -> Array(x.grad), vars)
    y2 = Array(y2.data)

    if !all(d -> abs(d) < eps, y1 - y2)
        throw("output of CPU and CUDA mismatch")
    end
    foreach(zip(gxs1, gxs2)) do g
        diff = g[1] - g[2]
        if !all(d -> abs(d) < eps, diff)
            throw(diff)
        end
    end

    for v in vars
        v.data = Array(v.data)
        v.grad = zeros(v.data)
    end
    true
end
