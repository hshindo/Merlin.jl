export checkgrad

function checkgrad(f, args...; eps=1e-3)
    vars = filter(a -> isa(a,Var), args)
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
    for i = 1:length(gxs1)
        diff = gxs1[i] - gxs2[i]
        all(d -> abs(d) < eps, diff) && continue
        println(diff)
        return false
    end
    true
end

function checkgrad2(f::Function, args::Tuple; eps=1e-3)
    args = map(a -> Var(a,:cpu), args)
    foreach(zerograd!, args)
    y = f()
    gradient!(y)
    gxs1 = map(x -> x.grad, args)
    gxs2 = map(args) do arg
        x = arg.data
        gx = similar(x)
        for k = 1:length(x)
            xk = x[k]
            x[k] = xk + eps
            y1 = f().data
            x[k] = xk - eps
            y2 = f().data
            x[k] = xk
            gx[k] = sum(y1 - y2) / 2eps
        end
        gx
    end
    all(1:length(gxs1)) do i
        checkdiff(gxs1[i], gxs2[i], eps)
    end
end

function testcuda(f::Function, args::Tuple; eps=1e-3)
    args = map(a -> Var(a,:cpu), args)
    foreach(zerograd!, args)
    y1 = f()
    gradient!(y1)
    gxs1 = map(x -> x.grad, args)

    args = map(a -> Var(a,:cuda), args)
    foreach(zerograd!, args)
    y2 = f()
    gradient!(y2)
    gxs2 = map(x -> Array(x.grad), args)

    testdiff(y1.data, Array(y2.data), eps)
    all(1:length(gxs1)) do i
        testdiff(gxs1[i], gxs2[i], eps)
    end
end
