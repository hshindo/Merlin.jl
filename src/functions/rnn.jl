function rnn{T}(x::CuArray{T}, hsize::Int, nlayers::Int)
    h = CUDNN.handle(x)
    desc = CUDNN.RNNDesc()

    dropoutdesc
    cudnnSetRNNDescriptor(desc, hsize, nlayers, dropoutdesc)

    cudnnRNNForwardInference(h, desc, seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,
    wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes)
end
