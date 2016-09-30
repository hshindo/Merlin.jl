export lrn

"""
    lrn(x, n, k, alpha, beta)

Local Response Normalization (LRN).

* x::Var: 4-d Var.
* n::Int: Normalization window width.
* k::Float64: Smoothing parameter.
* alpha::Float64: Normalizer scaling parameter.
* beta::Float64: Normalizer power parameter.
"""
function lrn(x::Var, n::Int, k::Float64, alpha::Float64, beta::Float64)
    y = lrn(x.data, n, k, alpha, beta)
    df(gy) = hasgrad(x) && ∇lrn!(x.data, x.grad, y.data, y.grad)
    Var(y, [x], lrn, df)
end

function lrn{T}(x::Array{T,4}, n::Int, k::Float64, alpha::Float64, beta::Float64)
    scale = zeros(Float64, size(x)) 
    winsize = size(x,1) * size(x,2)
    batoffs = div(length(x), size(x,4))

    padsqx = zeros(Float64, winsize * (size(x,3)+n-1) * size(x,4))
    padpreoffs = winsize * div(n-1, 2) 
    padbatoffs = div(length(padsqx), size(x,4))
    
    for i=0:size(x,4)-1
        for j=1:batoffs
            padsqx[j + padpreoffs + i*padbatoffs] = (Float64(x[j + i*batoffs]))^2 * alpha / n
        end

        for j=1:winsize
            scale[j + i*batoffs] = k
        end

        for j1=0:n-1
            for j2=1:winsize
                scale[j2 + i*batoffs] += padsqx[j2 + j1*winsize + i*padbatoffs]
            end
        end

        for j1=1:size(x,3)-1
            for j2=1:winsize
                scale[j2 + j1*winsize + i*batoffs] = scale[j2 + (j1-1)*winsize + i*batoffs]
                    - padsqx[j2 + (j1-1)*winsize + i*padbatoffs]
                    + padsqx[j2 + (j1+n-1)*winsize + i*padbatoffs]
            end
        end
    end

    pwdscale = scale.^-beta
    y = similar(x)
    for i=1:length(x)
        y[i] = T(pwdscale[i]) * x[i]
    end
    y, scale, pwdscale
end

lrn(x::CuArray) = JuCUDNN.lrn(x)

function ∇lrn!{T}(x::Array{T,4}, gx::Array{T,4}, y::Array{T,4}, gy::Array{T,4},
    scale::Array{Float64,4}, pwdscale::Array{Float64,4}, n::Int, alpha::Float64,
    beta::Float64)

    gscale = similar(scale)
    winsize = size(x,1) * size(x,2)
    batoffs = div(length(x), size(x,4))

    padsqx = zeros(Float64, winsize * (size(x,3)+n-1) * size(x,4))
    padpreoffs = winsize * div(n-1, 2) 
    padbatoffs = div(length(padsqx), size(x,4))

    for i=0:size(x,4)-1
        for j=1:batoffs
            padsqx[j + padpreoffs + i*padbatoffs] = -2 * alpha / n * beta * (
                gy[j + i*batoffs] * y[j + i*batoffs] / scale[j + i*batoffs])
        end

        for j=1:winsize
            gscale[j + i*batoffs] = 0
        end

        for j1=0:n-1
            for j2=1:winsize
                gscale[j2 + i*batoffs] += padsqx[j2 + j1*winsize + i*padbatoffs]
            end
        end

        for j1=1:size(x,3)-1
            for j2=1:winsize
                gscale[j2 + j1*winsize + i*batoffs] = gscale[j2 + (j1-1)*winsize + i*batoffs]
                    - padsqx[j2 + (j1-1)*winsize + i*padbatoffs]
                    + padsqx[j2 + (j1+n-1)*winsize + i*padbatoffs]
            end
        end
    end

    for i=1:length(x)
        gx[i] += gy[i] * T(pwdscale[i]) + x[i] * T(gscale[i])
    end
end

∇lrn!(x::CuArray, gx, y, gy) = JuCUDNN.∇lrn!(x, y, gy, gx)
