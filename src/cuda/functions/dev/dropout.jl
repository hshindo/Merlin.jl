import Merlin: dropout. ∇dropout!

function dropout{T,N}(x::Var{CuArray{T,N}}, rate::Float64)
    y, work = CUDNN.dropout(x.data, rate)
    df(gy::CuArray) = isvoid(x.grad) || ∇dropout!(gy, x.grad, rate, work)
    Var(y, df, (x,))
end

function ∇dropout!()
end
