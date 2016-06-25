using CUDA
using CUDNN

@testset "cuda" for i = 1:5
  x = Var(rand(T,10,5))
  for f in [sigmoid, tanh]
    @test checkgrad(() -> f(x), x)
  end
end
