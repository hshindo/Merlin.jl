export lrn

function lrn(x::Var)
    y = lrn(x.data)
    df(gy) = hasgrad(x) && ∇lrn!(x.data, x.grad, y.data, y.grad)
    Var(y, [x], lrn, df)
end

function lrn(x::Array)
    throw("Not implemented yet.")
end

lrn(x::CuArray) = JuCUDNN.lrn(x)

function ∇lrn!(x::Array, gx::Array, y::Array, gy::Array)
    throw("Not implemented yet.")
end

∇lrn!(x::CuArray, gx, y, gy) = JuCUDNN.∇lrn!(x, y, gy, gx)
