






@testset "rnn" begin
    x = param(randn(T,20,10))
    for nlayers = 1:1
        #lstm = LSTM(T, 20, 15, nlayers, 0.0, true)
        #@test_grad lstm(x,[5,3,2]) x
        #@test_cuda lstm x batchdims
    end
end



@testset "pack" begin
    x = param(randn(T,10,10))
    @test_grad pack(x,[2,5,3],0) x
    @test_cuda pack(x,[2,5,3],0) x
end



@testset "split" begin
    x = param(rand(T,10,10))
    #@test_grad split x 2 [3,5,2]
end

@testset "standardize" begin
    x = param(randn(T,1,5)*3+2)
    #f = Standardize(T,size(x.data))
    #@testgrad f(x,true) x f.scale f.bias
end
