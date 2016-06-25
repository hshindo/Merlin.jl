@testset "gru" for i = 1:5
  x = Var(rand(T,100,1))
  h = Var(rand(T,100,1))
  f = GRU(T, 100)
  @test @gradcheck f(:x=>x, :h=>h) (x,h)
  #checkgrad(() -> f(:x=>x, :h=>h), x, h)
end
