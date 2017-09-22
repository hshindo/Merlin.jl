export @testgrad

using Base.Test

macro testgrad(f, vars...)
    vars = Expr(:tuple, vars...)
    f = Expr(:->, Expr(:tuple), f) # convert f to anonymouos function i.e., () -> f
    quote
        f = $(esc(f))
        vars = $(esc(vars))
        @test all(vars) do v
            checkgrad(f, v)
        end
    end
end

function checkgrad(f::Function, var::Var)
    check_eps = 1e-3
    var.grad = zeros(var.data)
    y = f()
    gradient!(y)
    gx1 = var.grad

    x = var.data
    gx2 = similar(x)
    for k = 1:length(x)
        xk = x[k]
        x[k] = xk + check_eps
        y1 = copy(f().data)
        x[k] = xk - check_eps
        y2 = copy(f().data)
        x[k] = xk
        gx2[k] = sum(y1-y2) / (2*check_eps)
    end
    diff = gx1 - gx2
    if maximum(abs,diff) >= check_eps
        println(x)
        println()
        println(gx1)
        println()
        println(gx2)
        println()
        println(diff)
        false
    else
        true
    end
end
