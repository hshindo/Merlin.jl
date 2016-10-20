const gradeps = 1e-2

function approx_grad(f, xs::Var...)
    map(xs) do v
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

function checkgrad(f, xs::Var...)
    for x in xs
        x.grad = zeros(x.data)
    end
    y = f()
    gradient!(y)
    approx_gxs = approx_grad(f, xs...)
    for i = 1:length(xs)
        gx1 = xs[i].grad
        gx2 = approx_gxs[i]
        all(d -> abs(d) < gradeps, gx1 - gx2) && continue
        println(gx1 - gx2)
        return false
    end
    true
end

function checkcuda(f, xs::Var...)
    eps = 1e-2
    for x in xs
        x.grad = zeros(x.data)
    end
    out = f()
    y = copy(out.data)
    gxs = map(v -> v.grad, gradient!(out))

    for x in xs
        x.data = CuArray(x.data)
        x.grad = zeros(x.data)
    end

    out = f()
    cuy = Array(out.data)
    cugxs = map(v -> Array(v.grad), gradient!(out))

    b = true
    for i = 1:length(gxs)
        diff = gxs[i] - cugxs[i]
        if any(d -> abs(d) >= eps, diff)
            println(diff)
            b = false
        end
    end
    for x in xs
        x.data = Array(x.data)
        x.grad = zeros(x.data)
    end
    b
end
