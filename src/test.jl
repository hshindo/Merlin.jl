using Base.Test

export test_gradient, test_backend

function test_gradient(f, xs...; tol=1e-3)
    params = filter(x -> isa(x,Var) && isparam(x), [xs...])
    y = f(xs...)

    foreach(zerograd!, params)
    gradient!(y)
    gxs1 = map(x -> x.grad, params)

    gxs2 = map(params) do p
        x = p.data
        gx2 = similar(x)
        for k = 1:length(x)
            xk = x[k]
            x[k] = xk + 1e-3
            y1 = copy(f(xs...).data) # In case y == x
            x[k] = xk - 1e-3
            y2 = copy(f(xs...).data)
            x[k] = xk
            gx2[k] = sum(y1-y2) / 2e-3
        end
        gx2
    end

    for (gx1,gx2) in zip(gxs1,gxs2)
        @test maximum(abs,gx1-gx2) <= tol
    end
end

function test_backend(backend, f, xs...; tol=1e-3)
    LibCUDA.Configured || return

    d_f = compile(f, backend)
    d_xs = map(x -> isa(x,Var) ? compile(x,backend) : x, xs)

    foreach(x -> isa(x,Var) && isparam(x) && zerograd!(x), xs)
    foreach(x -> isa(x,Var) && isparam(x) && zerograd!(x), d_xs)

    y = f(xs...)
    d_y = d_f(d_xs...)
    @test y.data ≈ d_y.data

    gradient!(y)
    gradient!(d_y)
    for (x,d_x) in zip(xs,d_xs)
        isa(x,Var) || continue
        isparam(x) || continue
        @test x.grad ≈ d_x.grad
    end
end
