function checkgrad(f, args...; eps=1e-2)
    xs = filter(a -> typeof(a) == Var && !isconst(a), args)
    foreach(zerograd!, xs)
    y = f(args...)
    gradient!(y)
    gxs1 = map(x -> x.grad, xs)
    gxs2 = map(xs) do v
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
    foreach(i -> checkdiff(gxs1[i], gxs2[i], eps), 1:length(gxs1))
    true
end

function checkcuda(f, args...; eps=1e-2)
    xs = filter(a -> typeof(a) == Var && !isconst(a), args)
    foreach(zerograd!, xs)
    y = f(args...)
    gradient!(y)
    gxs = map(x -> x.grad, xs)

    for x in xs
        x.data = CuArray(x.data)
        x.grad = zeros(x.data)
    end
    cuy = f(args...)
    gradient!(cuy)
    cugxs = map(x -> Array(x.grad), xs)
    checkdiff(y.data, Array(cuy.data), eps)
    foreach(i -> checkdiff(gxs[i], cugxs[i], eps), 1:length(gxs))

    for x in xs
        x.data = Array(x.data)
        x.grad = zeros(x.data)
    end
    true
end

function checkdiff{T}(x1::Array{T}, x2::Array{T}, eps)
    all(d -> abs(d) < T(eps), x1 - x2) && return nothing
    println(x1 - x2)
    throw("Check failed.")
end
