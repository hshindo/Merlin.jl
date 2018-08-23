const T = Float32

@testset "functions" for i = 1:5

@testset "activation" begin
    x = param(randn(T,10,5))
    for i = 1:length(x.data)
        abs(x.data[i]) < 0.1 && (x.data[i] += 1)
    end
    for f in (relu,sigmoid,tanh)
        @test_grad f(x) x
        @test_cuda f(x) x
    end
end

@testset "pack" begin
    x = param(randn(T,10,10))
    @test_grad pack(x,[2,5,3],0) x
end

@testset "reduction" begin
    x = param(randn(T,10,10)*T(10))
    @testset "max" begin
        for dim = 1:2
            #@test_grad max(x,dim) x
            #@test_cuda max(x,dim) x
        end
        #@test_grad max(x,[2,5,3]) x
        #@test_cuda max(x,[2,5,3]) x
    end
end

end
