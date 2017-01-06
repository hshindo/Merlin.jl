export checkgrad, checkcuda

function checkgrad(f, args...; eps=1e-3)
    vars = filter(a -> isa(a,Var) && !isa(a.grad,Void), args)
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
            y1 = f(args...).data
            x[k] = xk - eps
            y2 = f(args...).data
            x[k] = xk
            gx[k] = sum(y1 - y2) / 2eps
        end
        gx
    end
    foreach(zip(gxs1,gxs2)) do g
        diff = g[1] - g[2]
        all(d -> abs(d) < eps, diff) || throw(diff)
    end
    use_cuda && checkcuda(f, args..., eps=eps)
    true
end

function checkcuda(f::Function, args...; eps=1e-3)
    args = map(a -> isa(a,Var) ? Var(a,:cpu) : a, args)
    vars = filter(a -> isa(a,Var) && !isa(a.grad,Void), args)
    foreach(zerograd!, vars)
    y1 = f(args...)
    gradient!(y1)
    gxs1 = map(x -> x.grad, vars)

    args = map(a -> isa(a,Var) ? Var(a,:cuda) : a, args)
    vars = filter(a -> isa(a,Var) && !isa(a.grad,Void), args)
    foreach(zerograd!, vars)
    y2 = f(args...)
    gradient!(y2)
    gxs2 = map(x -> Array(x.grad), vars)

    all(d -> abs(d) < eps, y1.data - Array(y2.data)) || throw("output of CPU and CUDA mismatch")
    foreach(zip(gxs1, gxs2)) do g
        diff = g[1] - g[2]
        all(d -> abs(d) < eps, diff) || throw(diff)
    end
    true
end
