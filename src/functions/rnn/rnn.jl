export RNN

type RNN
    f
end

function Base.split{T,N}(x::Array{T,N})
    arrays = Array{T,N-1}[]
    dims = Base.front(size(x))
    for i = 1:size(x,N)
        a = unsafe_wrap(Array, pointer(x,i), dims)
        push!(arrays, a)
    end
end

function (rnn::RNN)(x::Var)
    ys = []
    for i = 1:length(xs)
        y = rnn.f(xs[i])
        push!(ys, y)
    end
    ys
end

#=
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
=#
