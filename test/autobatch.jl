workspace()
using Merlin

T = Float32
x = Var()
x = Linear(T,300,100)(x)
x = relu(x)

const T = Float32
data_x = [rand(T,300) for i=1:100]
data_y = [rand(1:5) for i=1:100]

for i = 1:length(data_x)
    x = Var(data_x[i])
    x = Linear(T,300,100)(x)
    x = relu(x)
    x = Linear(T,100,3)(x)
    z = softmax(x)
    y = data_y[i]
    l = crossentropy(y, z)
    loss += l
end
opt = SGD(0.001)
minimize(loss, opt, epoch=10)
