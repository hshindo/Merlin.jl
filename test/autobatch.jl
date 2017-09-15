workspace()
using Merlin

T = Float32
x = Var()
x = Linear(T,300,100)(x)
x = relu(x)

const T = Float32
data_x = [rand(T,300) for i=1:100]
data_y = [rand(1:5) for i=1:100]

x = Var(data_x[i])
h = Linear(T,300,100)(x)
h = relu(h)
h = Linear(T,100,3)(h)
z = softmax(h)
loss = crossentropy(data_y[i], z)
nn = compile(loss, x)

opt = SGD(0.001)
for epoch = 1:10
    minimize(nn, data_x, data_y, opt, minibatch=32)
end
