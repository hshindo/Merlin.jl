export GatedConv1D

struct GatedConv1D
    cnn::Conv1D
end

function GatedConv1D{T}(::Type{T}, filtersize::Int, outsize::Int, pad::Int, stride::Int)
    GatedConv1D(Conv1D(T, filtersize, 2outsize, pad, stride))
end

function (f::GatedConv1D)(x::Var)
    y = f.cnn(x)
    n = size(y.data, 1) ÷ 2
    y1 = y[1:n,:]
    y2 = y[n+1:2n,:]
    y1 .* sigmoid(y2)
end
