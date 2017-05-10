function setup_model()
    T = Float32

    x = Var()
    y = pairwise(x)
    #y = wordembeds(x)
    y = convolution(y, (50,), strides=(10,), pads=(20,))
    y = pooling(:max, y)
    charfun = compile(y)
end
