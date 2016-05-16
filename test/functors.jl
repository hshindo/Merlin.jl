const T = Float64

@testset "functors" for i = 1:5
  x = Var(rand(T,10,5))
  x1 = Var(rand(T,10,5))
  x2 = Var(rand(T,10,5))
  x3 = Var(rand(T,10,5))

  @testset "activation" begin
    @test checkgrad(Activation("relu"), x)
    @test checkgrad(Activation("tanh"), x)
    @test checkgrad(Activation("sigmoid"), x)
  end

  @test checkgrad(Concat(1), x1, x2, x3)
  @test checkgrad(Concat(2), x1, x2, x3)
  #@test checkgrad(Concat(3), x1, x2, x3)

  #@test checkgrad(Conv(5,(3,3),(1,1),(1,1)), x)

  @testset "crosentropy" begin
    p = Var(rand(1:10,size(x.val,2)))
    #@test checkgrad(CrossEntropy(), p, x)
  end

  @test checkgrad(Linear(T,10,7), x)
  @test checkgrad(LogSoftmax(), x)

  @test checkgrad(Add(), x1, x2)
  @test checkgrad(ElemAdd(), x1, x2)
  @test checkgrad(Subtract(), x1, x2)
  @test checkgrad(ElemSubtract(), x1, x2)
  @test checkgrad(Mult(), Var(rand(T,10,5)), Var(rand(T,5,3)))
  @test checkgrad(ElemMult(), x2, x3)

  @test checkgrad(Reshape(5,2,5), x)
  @test checkgrad(Softmax(), x)
end
