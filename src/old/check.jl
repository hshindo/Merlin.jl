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

function checkgrad(tol::Float64, f::Function, var::Var)
    var.grad = zeros(var.data)
    y = f()
    gradient!(y)
    gx1 = var.grad

    x = var.data
    gx2 = similar(x)
    for k = 1:length(x)
        xk = x[k]
        x[k] = xk + 1e-3
        y1 = copy(f().data)
        x[k] = xk - 1e-3
        y2 = copy(f().data)
        x[k] = xk
        gx2[k] = sum(y1-y2) / 2e-3
    end
    if maximum(abs,gx1-gx2) > tol
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
