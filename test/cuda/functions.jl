using LibCUDA

setdevice(0)

@testset "activation" for i = 1:5
    tol = 1e-3
    x = randn(T,10)
    for i = 1:length(x)
        abs(x[i]) < 0.1 && (x[i] += 1)
    end
    cux = cu(x)

    #@testgrad crelu(x) x
    #@testgrad elu(x) x
    #@testgrad leaky_relu(x) x
    y = relu(x)
    cuy = Array(relu(cux))
    #@test maximum(abs,y-cuy) < tol
    #@testgrad selu(x) x
    #sigmoid(cux)
    #@testgrad swish(x,beta) x beta
    #tanh(cux)

    #@test Array(sigmoid(cx)) sigmoid(x)
end

@testset "dropout" for i=1:5
    tol = 1e-3
    x = Var(randn(T,10))
    cux = Var(CuArray(x.data))
    cuy = dropout(cux, 0.5)
end

@testset "softmax" for i = 1:5
    tol = 1e-3
    x1 = randn(T,10) + T(3)
    x2 = randn(T,10,5) + T(3)
    for x in (x1,x2)
        y1 = softmax(x)
        y2 = Array(softmax(CuArray(x)))
        @test maximum(abs,y1-y2) < tol
    end
end
