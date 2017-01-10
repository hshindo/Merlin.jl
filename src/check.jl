export checkgrad, checkcuda

function checkgrad(f, args...; eps=1e-3)
    vars = collect(filter(a -> isa(a,Var) && !isa(a.grad,Void), args))
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
        all(d -> abs(d) < eps, diff) || throw(diff)
    end
    #use_cuda && checkcuda(f, args..., eps=eps)
    true
end

function checkcuda(f::Function, args...; eps=1e-3)
    args1 = map(a -> isa(a,Var) ? Var(a,:cpu) : a, args)
    vars1 = collect(filter(a -> isa(a,Var) && !isa(a.grad,Void), args1))
    foreach(zerograd!, vars1)
    y1 = f(args1...)
    gradient!(y1)
    gxs1 = map(x -> x.grad, vars1)

    args2 = map(a -> isa(a,Var) ? Var(a,:cuda) : a, args)
    vars2 = collect(filter(a -> isa(a,Var) && !isa(a.grad,Void), args2))
    foreach(zerograd!, vars2)
    y2 = f(args2...)
    gradient!(y2)
    gxs2 = map(x -> Array(x.grad), vars2)

    all(d -> abs(d) < eps, y1.data - Array(y2.data)) || throw("output of CPU and CUDA mismatch")
    foreach(zip(gxs1, gxs2)) do g
        diff = g[1] - g[2]
        if !all(d -> abs(d) < eps, diff)
            #println(g[1])
            #println(g[2])
            throw(diff)
        end
    end
    true
end
