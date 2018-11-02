using Test
export checkgrad

#=
macro checkgrad(f, params, atol=1e-3, cuda=true)
    quote
        checkgrad(() -> $(esc(f)), $(map(esc,params)...), atol=$atol, cuda=$cuda)
    end
end
=#

function checkgrad(f, inputs::Var...; atol=1e-3, cuda=true)
    params = tuple(filter(isparam, [inputs...])...)
    y = f()
    zerograd!.(params)
    gradient!(y)
    gxs1 = gradient.(params)
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
        foreach(x -> todevice!(x,0), inputs)
        d_y = f()
        zerograd!.(params)
        gradient!(d_y)
        d_gxs = gradient.(params)

        @test maximum(abs,y.data-Array(d_y.data)) <= atol
        for (gx,d_gx) in zip(gxs1,d_gxs)
            @test maximum(abs,gx-Array(d_gx)) <= atol
        end
        foreach(x -> todevice!(x,-1), inputs)
    end
end
