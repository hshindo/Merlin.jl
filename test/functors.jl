const T = Float64

@testset "add" for i = 1:5
  x1 = rand(T, 10, 5)
  x2 = rand(T, 10, 5)
  @test check_gradient(Add(), x1, x2)
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
  x = rand(T, 10, 5)
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

# lookup

# lookuplinear

#
