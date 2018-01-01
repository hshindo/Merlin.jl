const T = Float32

@testset "activation" for i = 1:5
    tol = 1e-3
    x = Var(randn(T,10,5))
    for i = 1:length(x.data)
        abs(x.data[i]) < 0.1 && (x.data[i] += 1)
    end
    beta = Var([T(1)])

    #@testgrad crelu(x) x
    #@testgrad elu(x) x
    #@testgrad leaky_relu(x) x
    #@test @checkgrad tol relu(x) x
    #@testgrad selu(x) x
    #@test @checkgrad tol sigmoid(x) x
    #@testgrad swish(x,beta) x beta
    #@test @checkgrad tol tanh(x) x
end

@testset "softmax" for i = 1:5
    x1 = Var(randn(T,10)+3)
    x2 = Var(randn(T,10,5)+3)
    for x in (x1,x2)
        @test @checkgrad softmax(x) x
        #@testgrad logsoftmax(x) x
    end
end
