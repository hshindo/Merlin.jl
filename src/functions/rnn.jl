mutable struct LSTM
    WU::Var
    b::Var
    h0::Var
    c0::Var
end

function (lstm::LSTM)(insize::Int, hsize::Int, nlayers::Int, droprate::Float64)
    W = cat(2, [init_W(T,insize,outsize) for i=1:4]...)
    U = cat(2, [init_U(T,outsize,outsize) for i=1:4]...)
    WU = cat(1, W, U)
    b = zeros(T, 4outsize)
    # b[outsize+1:2outsize] = -1
    #b[1:outsize] = ones(T, outsize) # forget gate initializes to 1
    h0 = zeros(T, outsize, 1)
    c0 = zeros(T, outsize, 1)
    
end
