const T = Float64

@testset "add" begin
  x1 = rand(T, 10, 5)
  x2 = rand(T, 10, 5)
  x3 = rand(T, 10, 1)
  @test check_gradient(Add(), x1, x2)
  @test check_gradient(ElemAdd(), x2, x3)
end

# blas


@testset "concat" for i = 1:5
  x1 = rand(T, 10, 5, 2)
  x2 = rand(T, 10, 5, 2)
  x3 = rand(T, 10, 5, 2)
  @test check_gradient(Concat(1), x1, x2, x3)
  @test check_gradient(Concat(2), x1, x2, x3)
  @test check_gradient(Concat(3), x1, x2, x3)
end

@testset "crosentropy" for i = 1:5
  p = [rand(1:10) for i=1:5]
  @test check_gradient(CrossEntropy(p), x)
end

@testset "linear" for i = 1:5
  x = rand(T, 10, 5)
  f = Linear(T, 10, 8)
  @test check_gradient(f, x)
end

@testset "logsoftmax" for i = 1:5
  x = rand(T, 10, 5)
  @test check_gradient(LogSoftmax(), x)
end

@testset "lookup" for i = 1:5
end

@testset "lookuplinear" for i = 1:5
end

@testset "max" for i = 1:5
  x = rand(T, 10, 5, 2)
  #@test check_gradient(Max(1), x)
  #@test check_gradient(Max(2), x)
  #@test check_gradient(Max(3), x)
end

@testset "maxpooling2d" for i = 1:5
end

@testset "multiply" for i = 1:5
  x1 = rand(T, 100, 50)
  x2 = rand(T, 50, 30)
  x3 = rand(T, 50, 30)
  @test check_gradient(Multiply(), x1, x2)
  @test check_gradient(ElemMultiply(), x2, x3)
end

@testset "relu" for i = 1:5
  x = rand(T, 10, 5)
  @test check_gradient(ReLU(), x)
end

@testset "reshape" for i = 1:5
  x = rand(T, 10, 5, 2)
  f = Reshape(5, 10, 2)
  @test check_gradient(f, x)
end

@testset "sigmoid" for i = 1:5
  x = rand(T, 10, 5)
  @test check_gradient(Sigmoid(), x)
end

@testset "tanh" for i = 1:5
  x = rand(T, 10, 5)
  @test check_gradient(Tanh(), x)
end

@testset "window2d" for i = 1:5
  x = rand(T, 100, 50)
  f = Window2D(50,2,1,1,5,5)
  @test check_gradient(f, x)
end
