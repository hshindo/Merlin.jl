const T = Float64

@testset "logsoftmax" begin
  x = rand(T, 10, 5)
  @test check_gradient(LogSoftmax(), x)
end

@testset "relu" begin
  x = rand(T, 10, 5)
  @test check_gradient(ReLU(), x)
end

@testset "sigmoid" begin
  x = rand(T, 10, 5)
  @test check_gradient(Sigmoid(), x)
end

@testset "softmax" begin
  x = rand(T, 10, 5)
  @test check_gradient(Softmax(), x)
end

@testset "tanh" begin
  x = rand(T, 10, 5)
  @test check_gradient(Tanh(), x)
end

@testset "crosentropy" begin
  p = [rand(1:10) for i=1:5]
  x = rand(T, 10, 5)
  #@test check_gradient(CrossEntropy(), p, x)
end

@testset "add" begin
  x1 = rand(T, 10, 5)
  x2 = rand(T, 10, 5)
  x3 = rand(T, 10, 1)
  @test check_gradient(Add(), x1, x2)
  @test check_gradient(ElemAdd(), x1, x2)
  @test check_gradient(ElemAdd(), x2, x3)
end

@testset "multiply" begin
  x1 = rand(T, 100, 50)
  x2 = rand(T, 50, 30)
  x3 = rand(T, 50, 30)
  @test check_gradient(Multiply(), x1, x2)
  @test check_gradient(ElemMultiply(), x2, x3)
end

@testset "subtract" begin
  x1 = rand(T, 10, 5)
  x2 = rand(T, 10, 5)
  x3 = rand(T, 10, 1)
  @test check_gradient(Subtract(), x1, x2)
  @test check_gradient(ElemSubtract(), x1, x2)
  @test check_gradient(ElemSubtract(), x2, x3)
end

@testset "concat" for i = 1:5
  x1 = rand(T, 10, 5, 2)
  x2 = rand(T, 10, 5, 2)
  x3 = rand(T, 10, 5, 2)
  @test check_gradient(Concat(1), x1, x2, x3)
  @test check_gradient(Concat(2), x1, x2, x3)
  @test check_gradient(Concat(3), x1, x2, x3)
end

@testset "linear" for i = 1:5
  x = rand(T, 10, 5)
  f = Linear(T, 10, 8)
  @test check_gradient(f, x)
end

@testset "lookup" for i = 1:5
end

@testset "lookuplinear" for i = 1:5
end

@testset "max" for i = 1:5
  x = rand(Float64, 10, 5, 2)
  #@test check_gradient(Max(1), x)
  #@test check_gradient(Max(2), x)
  #@test check_gradient(Max(3), x)
end

@testset "maxpooling2d" for i = 1:5
end

@testset "reshape" for i = 1:5
  x = rand(T, 10, 5, 2)
  f = Reshape(5, 10, 2)
  #@test check_gradient(f, x)
end

@testset "window2d" for i = 1:5
  x = rand(T, 10, 5)
  f = Window2D(10,2,1,1)
  @test check_gradient(f, x)
end
