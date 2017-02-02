function setup_model()
    T = Float32
    h = 1000
    fs = [Linear(T,784,h), Linear(T,h,h), Linear(T,h,10)]
    (x::Var, y=nothing) -> begin
        x = fs[1](x)
        x = relu(x)
        x = fs[2](x)
        x = relu(x)
        x = fs[3](x)
        y == nothing ? argmax(x.data,1) : crossentropy(y,x)
    end
end
