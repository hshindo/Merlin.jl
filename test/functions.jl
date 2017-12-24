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
    @test @checkgrad tol relu(x) x
    #@testgrad selu(x) x
    @test @checkgrad tol sigmoid(x) x
    #@testgrad swish(x,beta) x beta
    @test @checkgrad tol tanh(x) x
end
