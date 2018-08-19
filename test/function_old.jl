










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
