export dropout

"""
    dropout(x::Var, rate::Float64)

This is inteded to be used only for training.
For testing, omit the dropout function.
"""
function dropout(x::Var, rate::Float64)
    if config.train
        T = eltype(x.data)
        rate = T(rate)
        rx = rand(T, length(x.data))
        data = dropout(x.data, rate, rx)
    else
        data = x.data
        rx = nothing
    end
    Var(data, x.batchdims, dropout, (x,rate), work=rx)
end
dropout(x::Node, rate::Float64) = Node(dropout, x, rate)

function dropout{T}(x::Array{T}, rate::T, rx::Vector{T})
    scale = T(1 / (1-rate))
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = ifelse(rx[i] <= rate, T(0), scale*x[i])
    end
    y
end

function addgrad!(y::Var, ::typeof(dropout), x::Var, rate)
    isvoid(x.grad) && return
    ∇dropout!(y.grad, x.grad, rate, y.work)
end

function ∇dropout!{T}(gy::Array{T}, gx::Array{T}, rate::T, rx::Vector{T})
    scale = T(1 / (1-rate))
    @inbounds for i = 1:length(gx)
        gx[i] += ifelse(rx[i] <= rate, T(0), scale*gy[i])
    end
end

#=
function forward{T}(::typeof(dropout), x::CuArray{T}, rate::Float64)
    h = CUDNN.handle(x)
    y = similar(x)
    p = Cint[0]
    cudnnDropoutGetStatesSize(h, p)
    statessize = p[1]
    states = CuArray{Int8}(Int(statessize))

    xdesc = CUDNN.TensorDesc(x)
    p = Cint[0]
    cudnnDropoutGetReserveSpaceSize(xdesc, p)
    reservesize = p[1]
    reservespace = CuArray{Int8}(Int(reservesize))

    desc = CUDNN.DropoutDesc()
    cudnnSetDropoutDescriptor(desc, h, rate, states, statessize, 0)
    cudnnDropoutForward(h, desc, xdesc, x, xdesc, y, reservespace, reservesize)

    function backward!(dy, dx)
        isvoid(dx) || cudnnDropoutBackward(h, desc, xdesc, dy, xdesc, dx, reservespace, reservesize)
    end
    y, backward!
end
=#
