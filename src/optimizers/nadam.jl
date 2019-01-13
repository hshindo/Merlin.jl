export Nadam

"""
    Nadam

Adam Optimizer

# References
* Kingma and Ba, ["Incorporating Nesterov Momentum into Adam"](http://arxiv.org/abs/1412.6980v8), ICLR 2015.
"""
mutable struct Nadam
    alpha::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
    cumbeta1::Float64
    states::IdDict
end

Nadam(alpha=0.001)= Nadam(alpha, 0.9, 0.999, 1e-8, 1.0, IdDict())

function (opt::Nadam)(x::CuArray{T}, gx::CuArray{T}) where T
    @assert length(x) == length(gx)
    if haskey(opt.states, x)
        m, v, t = opt.states[x]
    else
        m, v, t = zero(x), zero(x), 1
    end

    nadam_update1!(gx, m, v, opt.beta1, opt.beta2)
    beta1_t = opt.beta1 * (1 - 0.5 * 0.96^(t/250))
    beta1_u = opt.beta1 * (1 - 0.5 * 0.96^((t+1)/250))
    opt.cumbeta1 *= beta1_t
    coef_g = (1 - beta1_t) / (1 - opt.cumbeta1)
    coef_m = beta1_u / (1 - opt.cumbeta1*beta1_u)
    coef_v = 1 / (1 - opt.beta2^t)
    nadam_update2!(x, gx, m, v, coef_g, coef_m, coef_v, opt.eps, opt.alpha)

    opt.states[x] = (m, v, t+1)
    fill!(gx, T(0))
end
(opt::Nadam)(v::Var) = opt(v.data, v.grad)

@generated function nadam_update1!(g::CuArray{T}, m::CuArray{T}, v::CuArray{T}, beta1, beta2) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void nadam_update1(int n, $Ct *g, $Ct *m, $Ct *v, $Ct beta1, $Ct beta2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        m[idx] = beta1 * m[idx] + (1-beta1) * g[idx];
        v[idx] = beta2 * v[idx] + (1-beta2) * g[idx]*g[idx];
    }""")
    quote
        gdims, bdims = cudims(length(g))
        $k(gdims, bdims, length(g), pointer(g), pointer(m), pointer(v), T(beta1), T(beta2))
    end
end

@generated function nadam_update2!(x::CuArray{T}, g::CuArray{T}, m::CuArray{T}, v::CuArray{T},
    coef_g, coef_m, coef_v, eps, alpha) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void nadam_update2(int n, $Ct *x, $Ct *g, $Ct *m, $Ct *v,
        $Ct coef_g, $Ct coef_m, $Ct coef_v, $Ct eps, $Ct alpha) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        $Ct mm = coef_g * g[idx] + coef_m * m[idx];
        $Ct vv = coef_v * v[idx];
        x[idx] -= alpha * mm / (sqrt(vv) + eps);
    }""")
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, length(x), pointer(x), pointer(g), pointer(m), pointer(v),
            T(coef_g), T(coef_m), T(coef_v), T(eps), T(alpha))
    end
end
