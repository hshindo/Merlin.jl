T = Float64

@testset "gru" begin
  x = rand(T, 100, 1)
  h = rand(T, 100, 1)
  f = GRU(T, 100)
  @test check_gradient(f, x, h)
end
