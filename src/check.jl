using Test
export checkgrad

macro checkgrad(f, params, atol=1e-3, cuda=true)
    quote
        checkgrad(() -> $(esc(f)), $(map(esc,params)...), atol=$atol, cuda=$cuda)
    end
end

function checkgrad(f, params::Var...; atol=1e-3, cuda=true)
    setcpu()
    y = f()
    foreach(zerograd!, params)
    gradient!(y)
    gxs1 = map(x -> x.grad, params)
    gxs2 = map(params) do p
        x = p.data
        gx2 = similar(x)
        for k = 1:length(x)
            xk = x[k]
            x[k] = xk + 1e-3
            y1 = copy(f().data) # In case y == x
            x[k] = xk - 1e-3
            y2 = copy(f().data)
            x[k] = xk
            gx2[k] = sum(y1-y2) / 2e-3
        end
        gx2
    end

    for (gx1,gx2) in zip(gxs1,gxs2)
        @test maximum(abs,gx1-gx2) <= atol
    end

    if cuda && CUDA.AVAILABLE
        setcpu()
        y = f()
        foreach(zerograd!, params)
        gradient!(y)
        gxs = map(x -> copy(x.grad), params)

        setcuda()
        d_y = f()
        foreach(zerograd!, params)
        gradient!(d_y)
        d_gxs = map(x -> x.grad, params)

        @test maximum(abs,y.data-Array(d_y.data)) <= atol
        for (gx,d_gx) in zip(gxs,d_gxs)
            @test maximum(abs,gx-Array(d_gx)) <= atol
        end
        setcpu()
    end
end

macro checkcuda(f, params...)
    quote
        test_cuda(() -> $(esc(f)), $(map(esc,params)...))
    end
end

function checkcuda(f, params::Var...; atol=2e-3)
    CUDA.AVAILABLE || return
    setcpu()

    y = f()
    foreach(zerograd!, params)
    gradient!(y)
    gxs = map(x -> copy(x.grad), params)

    setcuda()
    d_y = f()
    foreach(zerograd!, params)
    gradient!(d_y)
    d_gxs = map(x -> x.grad, params)

    @test y.data ≈ Array(d_y.data) atol=atol
    for (gx,d_gx) in zip(gxs,d_gxs)
        @test gx ≈ Array(d_gx) atol=atol
    end

    setcpu()
end

mutable struct Trainer
    xs::Vector
    ys::Vector
    nepochs::Int
    lossfun

end

function fit()
end
