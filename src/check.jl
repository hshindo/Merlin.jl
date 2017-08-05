export @testgrad

using Base.Test

macro testgrad(eps, f, vars...)
    vars = Expr(:tuple, vars...)
    f = Expr(:->, Expr(:tuple), f) # convert f to anonymouos function i.e., () -> f
    quote
        eps = $(esc(eps))
        f = $(esc(f))
        vars = $(esc(vars))
        @test all(vars) do v
            checkgrad(eps, f, v)
        end
    end
end

function checkgrad(eps::Float64, f::Function, var::Var)
    zerograd!(var)
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
    diff = gx1 - gx2
    if maximum(abs,diff) >= eps
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
