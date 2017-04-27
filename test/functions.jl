const T = Float32

@testset "functions" for i = 1:5

# activation
x = Var(rand(T,10,5))
relu(x)
clipped_relu(x)
@test checkgrad(()->sigmoid(x), x)
@test checkgrad(()->tanh(x), x)

# argmax
x = Var(rand(T,10,7,5,3))
for dim = 1:ndims(x.data)
    y = argmax(x, dim)
end

# blas
A = Var(rand(T,10,5))
B = Var(rand(T,5,10))
x = Var(rand(T,5))
@test checkgrad(()->gemv('N',0.3,A,x), A, x)
@test checkgrad(()->gemv('T',0.3,B,x), B, x)

x1 = Var(rand(T,10,5))
x2 = Var(rand(T,5,10))
x3 = Var(rand(T,10,5))
@test checkgrad(()->gemm('N','N',0.2,x1,x2), x1, x2)
@test checkgrad(()->gemm('N','T',0.3,x1,x3), x1, x3)
@test checkgrad(()->gemm('T','N',0.4,x1,x3), x1, x3)
@test checkgrad(()->gemm('T','T',0.5,x1,x2), x1, x2)

# cat
x1 = Var(rand(T,10,5,2))
x2 = Var(rand(T,10,5,2))
x3 = Var(rand(T,10,5,2))
for dim = 1:ndims(x1.data)+1
    @test checkgrad(()->cat(dim,x1,x2,x3), x1, x2, x3)
end

# crossentropy
p = Var(rand(0:10,1,5))
q = Var(rand(T,10,5))
@test checkgrad(()->crossentropy(p,q), q)

# getindex
x = Var(rand(T,10,5))
for i in ((1:3,5:5), (Colon(),2:5))
    @test checkgrad(()->x[i...], x)
end

# linear
x = Var(rand(T,10,10))
f = Linear(T, 10, 7)
@test checkgrad(()->f(x), x, f.w, f.b)

# lookup
x = Var(rand(1:10000,10,5))
f = Lookup(T, 10000, 100)
y = f(x)

# lstm
x = Var(rand(T,30,20))
f = LSTM(T, 30, 10)
h = Var(zeros(T,10))
c = Var(zeros(T,10))
f(x)
@test checkgrad(()->f(x,h,c), x, f.w, f.b)
# bilstm
x = Var(rand(T,50,20))
f = BiLSTM(T, 50, 30)
#@test checkgrad(()->f(x), x, f.fw.w, f.fw.b, f.bw.w, f.bw.b)

# math
x = Var(rand(T,10,5)+1)
for op in (exp,log,transpose)
    @test checkgrad(()->op(x), x)
end

x1 = Var(rand(T,10,5))
x2 = Var(rand(T,10,5))
for op in (+, -, .*)
    @test checkgrad(()->op(x1,x2), x1, x2)
end
@test checkgrad(()->-x1, x1)
x3 = Var(rand(T,10,1))
x4 = Var(rand(T,1,5))
for op in (.+, .-, .*)
    @test checkgrad(()->op(x1,x3), x1, x3)
    @test checkgrad(()->op(x1,x4), x1, x4)
end
x5 = Var(rand(T,5,7))
x6 = Var(rand(T,5))
@test checkgrad(()->x1*x5, x1, x5)
@test checkgrad(()->x1*x6, x1, x6)

# reduce
x = Var(rand(T,10,5))
for dim = 1:ndims(x.data)
    max(x, dim)
    @test checkgrad(()->sum(x,dim), x)
end

# reshape
x = Var(rand(T,5,4,3))
@test checkgrad(()->reshape(x,3,4,5), x)
@test checkgrad(()->reshape(x,4,3,5), x)

# softmax
x1 = Var(rand(T,10)+1)
x2 = Var(rand(T,10,5)+1)
for x in (x1,x2)
    @test checkgrad(()->softmax(x), x)
    @test checkgrad(()->logsoftmax(x), x, eps=1e-2)
end

# window
x = Var(rand(T,10,5))
@test checkgrad(()->window(x,(10,)), x)

end
