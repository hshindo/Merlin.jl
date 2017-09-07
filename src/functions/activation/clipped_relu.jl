export clipped_relu

doc"""
    clipped_relu(x)

Clipped Rectified Linear Unit.
"""
clipped_relu(x::Var) = Var(clipped_relu(x.data), clipped_relu, (x,))

clipped_relu(x::Node) = Node(clipped_relu, x)

function clipped_relu{T}(x::Array{T})
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    y
end

function addgrad!(y::Var, ::typeof(clipped_relu), x::Var)
    isvoid(x.grad) || ∇clipped_relu!(y.grad, x.data, x.grad)
end

function ∇clipped_relu!{T}(gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(x)
        gx[i] += ifelse(T(0) < x[i] < T(20), gy[i], T(0))
    end
end
