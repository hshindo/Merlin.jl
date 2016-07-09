const T = Float64

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

  x = Data(rand(T,5,4))
  for f in [sigmoid, tanh]
    @test @checkgrad f(x) [x]
    #@test @cuda_test f(x) (x,)
  end

  x1 = Data(rand(T,10,5,2))
  x2 = Data(rand(T,10,5,2))
  x3 = Data(rand(T,10,5,2))
  for dim = 1:3
    @test @checkgrad concat(dim,x1,x2,x3) [x1,x2,x3]
  end

  x = Data(rand(T,10,5))
  l = Linear(T, 10, 7)
  #l.b = Data(rand(T, size(l.b.value)))
  @test @checkgrad l(x) [l.w,x]

  #x = Var(rand(1:10,3,2))
  #f = Lookup(Float32, 10, 5)
  #args = f(x).args
  #@test @gradcheck f(x) (x,)

  x1 = Data(rand(T,10,5))
  x2 = Data(rand(T,10,5))
  x3 = Data(rand(T,10,1))
  for op in [+,-,.*]
    @test @checkgrad op(x1,x2) [x1,x2]
    @test @checkgrad op(x1,x3) [x1,x3]
    @test @checkgrad op(x3,x1) [x3,x1]
  end
  x4 = Data(rand(T,5,7))
  @test @checkgrad *(x1,x4) [x1,x4]

  #x = Var(rand(T,10,5))
  #@test @gradcheck reshape(x,2,5,5) [x]

  x = Data(rand(T,10,5,3,4))
  for dim = 1:ndims(x.y)
    @test @checkgrad softmax(x,dim) [x]
    @test @checkgrad logsoftmax(x,dim) [x]
  end

  p = Data([1:5;])
  x = Data(rand(Float32,10,5))
  #@test @checkgrad softmax_crossentropy(p,x,1) [x]

  x = Data(rand(T,10,5,4,3))
  for dim = 1:ndims(x.y)
    @test @checkgrad sum(x,dim) [x]
  end
end
