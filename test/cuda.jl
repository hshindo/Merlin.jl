using JuCUDA
using JuCUDNN

@testset "cuda" for i = 1:5

x = Var(rand(T,5,4))
cux = Var(CuArray(x.data))
for f in [sigmoid, tanh]
    @test f(x).data 
end

end
