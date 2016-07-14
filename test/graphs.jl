@testset "gru" for i = 1:5
  x = Data(rand(T,100,1))
  h = Data(rand(T,100,1))
  f = GRU(T, 100)
  @test @checkgrad f(:x=>x, :h=>h) [x,h]
end
