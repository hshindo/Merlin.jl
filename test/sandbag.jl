workspace()
using Merlin
using Merlin.Caffe
using JuCUDA
using Base.LinAlg.BLAS
using Base.Test
using HDF5
using Compat

x1 = curand(Float32,4,3)
x1 = zeros(x1)
x2 = curand(Float32,4,3)
x2 = ones(x2)
x3 = zeros(x1)
y = concat(1,[x1,x2,x3])
Array(y)

x = Var(rand(Float32,5,4))
cux = Var(CuArray(x.data))
sigmoid(x)

type XXX
    dim::Int
    function XXX(dim)
        println("ok")
        new(dim)
    end
end

macro aaa(expr)
    expr.args[2] = eval(expr.args[2])
    quote
        function sss()
            $expr
        end
    end
end

q = quote begin
    @aaa begin
        d = 3
        x = XXX(d)
    end
end

q()

s = @aaa(1+2)
s()

c = macroexpand(:(@aaa(XXX(3)(5))))
c.args[2].args[1].args


nprocs()
path = "C:/Users/hshindo/Desktop/nin_imagenet.caffemodel"
g = Caffe.load(path)
g.nodes

function bench()
    for i = 1:10000
        @simd for j = 1:10000
            a = rand(Float32)
        end
        #rand(Float32,100,100)
        #a * b
    end
end
@time bench()
