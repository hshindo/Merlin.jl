using Base.Test
using Merlin

function gradient_check()

end

x1 = rand(Float32, 10, 5)
x2 = rand(Float32, 7, 5)
f = Concat(1)
y = f([x1,x2])
@test y.value == c
