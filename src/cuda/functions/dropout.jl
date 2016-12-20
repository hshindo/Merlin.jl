import Merlin.dropout

function dropout{T,N}(x::Var{CuArray{T,N}}, rate::Float64)
    y, work = CUDNN.dropout(x.data, rate)
    df(gy::CuArray) = x.grad == nothing || ∇dropout!(gy, x.grad, rate, work)
    Var(y, df, (x,))
end
