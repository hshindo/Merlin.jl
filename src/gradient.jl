using Test
export gradient!, topsort, checkgrad

"""
    topsort(tops::T...)

Topological sort.
"""
function topsort(top::T) where T
    sorted = T[]
    dict = IdDict{T,T}()
    function visit(x::T)
        haskey(dict,x) && return
        dict[x] = x
        for arg in x.args
            isa(arg,T) && visit(arg)
        end
        push!(sorted, x)
    end
    visit(top)
    sorted
end

"""
    gradient!(top::Var)

Compute gradients.
"""
function gradient!(top::Var)
    sorted = topsort(top)
    if isnothing(top.grad)
        top.grad = fill!(similar(top.data), 1)
    end
    for v in sorted
        if !isempty(v.args) && isnothing(v.grad)
            v.grad = fill!(similar(v.data), 0)
        end
    end
    for i = length(sorted):-1:1
        y = sorted[i]
        isnothing(y.grad) && continue
        isempty(y.args) && continue
        y.f(y, y.args...)
    end
    sorted
end

#=
macro checkgrad(f, params, atol=1e-3, cuda=true)
    quote
        checkgrad(() -> $(esc(f)), $(map(esc,params)...), atol=$atol, cuda=$cuda)
    end
end
=#

function checkgrad(f, inputs::Var...; atol=2e-3, cuda=true)
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

    if cuda && CUDA_AVAILABLE
        setdevice(0)
        foreach(todevice!, inputs)
        d_y = f()
        zerograd!.(params)
        gradient!(d_y)
        d_gxs = gradient.(params)

        @test maximum(abs,y.data-Array(d_y.data)) <= atol
        for (gx,d_gx) in zip(gxs1,d_gxs)
            if maximum(abs,gx-Array(d_gx)) > atol
                println(gx)
                println(Array(d_gx))
            end
            @test maximum(abs,gx-Array(d_gx)) <= atol
        end
        setdevice(-1)
        foreach(todevice!, inputs)
    end
end
