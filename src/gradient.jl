export checkgrad, checkcuda

function checkgrad(f, args::Var...; eps=1e-3, cuda=true)
    foreach(zerograd!, args)
    y = f()
    gradient!(y)
    gxs1 = map(x -> x.grad, args)
    gxs2 = map(args) do arg
        x = arg.data
        gx = similar(x)
        for k = 1:length(x)
            xk = x[k]
            x[k] = xk + eps
            y1 = f().data
            x[k] = xk - eps
            y2 = f().data
            x[k] = xk
            gx[k] = sum(y1 - y2) / 2eps
        end
        gx
    end
    foreach(i -> checkdiff(gxs1[i],gxs2[i],eps), 1:length(gxs1))
    #cuda && Pkg.installed("CUDA") != nothing && checkcuda(f, args...)
    true
end

function checkcuda(f, args::Var...; eps=1e-3)
    for x in args
        x.data = Array(x.data)
        x.grad = zeros(x.data)
    end
    y = f()
    gradient!(y)
    gxs = map(x -> x.grad, args)

    for x in args
        x.data = CuArray(x.data)
        x.grad = zeros(x.data)
    end
    cuy = f()
    gradient!(cuy)
    cugxs = map(x -> Array(x.grad), args)
    checkdiff(y.data, Array(cuy.data), eps)
    for i = 1:length(gxs)
        checkdiff(gxs[i],cugxs[i],eps)
    end

    for x in args
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
