const T = Float32

macro cuda_test(f, args)
  quote
    haskey(ENV, "USE_CUDA") || return true

    local f() = $(esc(f))
    local args = $(esc(args))
    eps = 1e-3
    y1 = f().value
    for v in args
      v.value = CuArray(v.value)
    end
    y2 = Array(f().value)
    b = all(d -> abs(d) < 2eps, y1 - y2)

    for v in args
      v.value = Array(v.value)
    end
    b
  end
end

@testset "functions" for i = 1:5

  x = Var(rand(T,5,4))
  for f in [sigmoid, tanh]
    @test @gradcheck f(x) [x]
    #@test @cuda_test f(x) (x,)
  end

  x1 = Var(rand(T,10,5,2))
  x2 = Var(rand(T,10,5,2))
  x3 = Var(rand(T,10,5,2))
  for dim = 1:3
    @test @gradcheck concat(dim,x1,x2,x3) [x1,x2,x3]
  end

  x = Var(rand(T,10,5))
  f = Linear(T, 10, 7)
  f.b = param(rand(T, size(f.b.value)))
  @test @gradcheck f(x) [f.w,x]

  x = Var(rand(T,10,5))
  #@test @gradcheck logsoftmax(x,2) (x,) # numerical error?

  x = Var(rand(1:10,3,2))
  f = Lookup(Float32, 10, 5)
  #args = f(x).args
  #@test @gradcheck f(x) (x,)

  x1 = Var(rand(T,10,5))
  x2 = Var(rand(T,10,5))
  x3 = Var(rand(T,10,1))
  for op in [+,-,.*]
    @test @gradcheck op(x1,x2) [x1,x2]
    @test @gradcheck op(x1,x3) [x1,x3]
    @test @gradcheck op(x3,x1) [x3,x1]
  end
  x4 = Var(rand(T,5,7))
  @test @gradcheck *(x1,x4) [x1,x4]

  x = Var(rand(T,10,5))
  @test @gradcheck reshape(x,2,5,5) [x]

  x = Var(rand(T,10,5))
  @test @gradcheck softmax(x,1) [x]

  p = Var([1:5;])
  x = Var(rand(Float32,10,5))
  @test @gradcheck softmax_crossentropy(p,x,1) [x]

  x = Var(rand(T,10,5,4,3))
  for dim = 1:ndims(x.value)
    @test @gradcheck sum(x,dim) [x]
  end
end
