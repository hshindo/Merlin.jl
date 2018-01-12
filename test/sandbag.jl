using Merlin
using LibCUDA

T = Float32
x = zerograd(randn(T,10,2))
f = LSTM(T, 10, 1, 0.0)
y = f(x, [2])
cux = compile(x, CUDABackend(0))
cuf = compile(f, CUDABackend(0))
cuy = cuf(cux, [2])
println(size(y))
println(y.data)
println("---")
println(cuy)
