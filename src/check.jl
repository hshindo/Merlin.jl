export @checkgrad

macro checkgrad(tol, f, vars...)
    vars = Expr(:tuple, vars...)
    f = Expr(:->, Expr(:tuple), f) # convert f to anonymouos function i.e., () -> f
    quote
        tol = $(esc(tol))
        f = $(esc(f))
        vars = $(esc(vars))
        all(v -> checkgrad(tol,f,v), vars)
    end
end

function checkgrad(tolerance::Float64, f::Function, var::Var)
    eps = 1e-3
    var.grad = zeros(var.data)
    y = f()
    gradient!(y)
    gx1 = var.grad

    x = var.data
    gx2 = similar(x)
    for k = 1:length(x)
        xk = x[k]
        x[k] = xk + eps
        y1 = copy(f().data)
        x[k] = xk - eps
        y2 = copy(f().data)
        x[k] = xk
        gx2[k] = sum(y1-y2) / 2eps
    end
    if maximum(abs,gx1-gx2) > tolerance
        println("x:")
        println(x)
        println("gx1:")
        println(gx1)
        println("gx2:")
        println(gx2)
        println("diff:")
        println(diff)
        false
    else
        true
    end
end

function checkgpu(f, xs...)
    y = f(x)
    cux =
    cuy = f(cux)
end
