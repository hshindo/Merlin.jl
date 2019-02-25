export WeightNorm

mutable struct WeightNorm <: Functor
    g::Var
end

function WeightNorm(::Type{T}) where T
    g = Fill(1)(T, 1, 1)
    WeightNorm(parameter(g))
end

function (wn::WeightNorm)(f::Functor)
    W = normalize(f.W, 2, dims=1) .* wn.g
    W = dropout(W, 0.33)
    Conv1d(f.ksize, f.padding, f.stride, f.dilation, f.ngroups, W, f.b)
end
