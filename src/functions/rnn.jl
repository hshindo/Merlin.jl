type RNN
    w
end

function (rnn::RNN){T}(x::CuArray{T}, hsize::Int, nlayers::Int)
    h = CUDNN.handle(x)
    desc = CUDNN.RNNDesc(h, seqlength)



    cudnnRNNForwardInference(h, desc, seqlength, xdesc, x, hxdesc, hx, cxdesc,cx,
        wdesc, w, ydesc, y, hydesc, hy, cydesc, cy, desc.workspace, length(desc.workspace))
    y
end
