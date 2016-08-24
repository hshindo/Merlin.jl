@testset "gru" for i = 1:5
  x = Var(rand(T,100,1)) |> zerograd!
  h = Var(rand(T,100,1)) |> zerograd!
  f = GRU(T, 100)
  @test checkgrad(()->f(x,h), x, h)
end
