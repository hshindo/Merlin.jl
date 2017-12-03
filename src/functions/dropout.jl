export dropout

doc"""
    dropout(x::Var, rate::Float64, train::Bool)

If `train` is true, drops elements randomly with probability ``rate`` and
scales the other elements by factor ``1 / (1 - rate)``.
Otherwise, it just returns `x`.
"""
function dropout(x::Var, rate::Float64, train::Bool)
    if train
        T = eltype(x)
        rx = rand(T, length(x.data))
        y = dropout(x.data, T(rate), rx)
        Var(y, x.batchdims, dropout, (x,rate,rx))
    else
        x
    end
end

dropout(x::Node, rate::Float64, train; name="") = Node(dropout, (x,rate,train), name)

function dropout(x::Array{T}, rate::T, rx::Vector{T}) where T
    scale = T(1 / (1-rate))
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = rx[i] <= rate ? T(0) : scale*x[i]
    end
    y
end

function addgrad!(y::Var, ::typeof(dropout), x::Var, rate, rx)
    isvoid(x.grad) || ∇dropout!(y.grad, x.grad, eltype(x.data)(rate), rx)
end

function ∇dropout!(gy::Array{T}, gx::Array{T}, rate::T, rx::Vector{T}) where T
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
