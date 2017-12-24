@testset "activation" for i = 1:5
    x = randn(T,10,5)
    for i = 1:length(x)
        abs(x[i]) < 0.1 && (x[i] += 1)
    end
    cux = cu(x)

    #@testgrad crelu(x) x
    #@testgrad elu(x) x
    #@testgrad leaky_relu(x) x
    relu(cux)
    #@testgrad selu(x) x
    sigmoid(cux)
    #@testgrad swish(x,beta) x beta
    tanh(cux)

    #@test Array(sigmoid(cx)) sigmoid(x)
end
