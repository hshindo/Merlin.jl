export lecun_tanh

doc"""
    tanh(x::Var)

Hyperbolic tangent function.
"""
Base.tanh(x::Var) = Var(tanh(x.data), ∇tanh!, (x,))
Base.tanh(x::Array) = tanh.(x)
Base.tanh(x::CuArray) = CUDNN.tanh(x)
Base.tanh(x::Node) = Node(tanh, (x,))

function ∇tanh!(y::Var, x::Var)
    isnothing(x.grad) && return
    ∇tanh!(y.data, y.grad, x.data, x.grad)
end

function ∇tanh!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end

∇tanh!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇tanh!(y, gy, x, gx)


lecun_tanh(x::Var) = Var(lecun_tanh(x.data), ∇lecun_tanh!, (x,))
lecun_tanh(x::Array) = lecun_tanh.(x)
lecun_tanh(x::T) where T<:AbstractFloat = T(1.17159tanh(2x/3))

@generated function lecun_tanh(x::CuArray{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void lecun_tanh($Ct *y, $Ct *x, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= length) return;
        y[idx] = 1.17159 * tanh(2/3 * x[idx]);
    }""")
    quote
        y = similar(x)
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(y), pointer(x), length(x))
        y
    end
end

function ∇lecun_tanh!(y::Var, x::Var)
    isnothing(x.grad) && return
    ∇lecun_tanh!(y.grad, x.data, x.grad)
end

function ∇lecun_tanh!(gy::CuArray{T}, x::CuArray{T}, gx::CuArray{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void grad_lecun_tanh($Ct *gy, $Ct *x, $Ct *gx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= length) return;
        gx[idx] += gy[idx] * 1.14393 * (1- tanh(2/3 * x[idx]));
    }""")
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(gy), pointer(x), pointer(gx), length(x))
    end
end
