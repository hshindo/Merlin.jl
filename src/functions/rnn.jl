type RNN
    w::Var
    b::Var
end

function RNN{T}(::Type{T}, xsize::Int, hsize::Int, droprate::Float64)
    w = uniform(T, -0.001, 0.001, hsize*4, xsize+hsize)
    u = orthogonal(T, hsize*4, xsize+hsize)
    w = cat(2, w, u)
    b = zeros(T,size(w,1))
    b[1:hsize] = ones(T, hsize) # forget gate initializes to 1
    RNN(zeerograd(w), zerograd(b))
end

function preprocess(xs::Vector)
    p = sortperm(xs, lt=(x,y)->isless(length(x),length(y)), rev=true)
    permute!(xs, p)
    x = transpose(cat(2,xs...))
    
end

function forward{T}(rnn::RNN, xs::Vector{CuVector{T}}, nlayers::Int)
    setbackend!(rnn.w, CuArray)
    setbackend!(rnn.b, CuArray)

    h = CUDNN.handle(x)
    rnndesc = CUDNN.RNNDesc(h, size(xs[1]))
    xdesc = map(x -> CUDNN.TensorDesc(x), xs)

    cudnnRNNForwardInference(h, desc, seqlength, xdesc, x, hxdesc, hx, cxdesc,cx,
        wdesc, w, ydesc, y, hydesc, hy, cydesc, cy, desc.workspace, length(desc.workspace))
    y
end
