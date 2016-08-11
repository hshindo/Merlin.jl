function lpnorm(x::Var, p)
    y = lpnorm(x.data, p)
    df(gy) = hasgrad(x) && ∇lpnorm!(x, p, x.grad, gy)
    Var(y, [x], lpnorm, df)
end

function lpnorm{T,P}(x::Array{T}, p::P)
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = x[i]^p
    end
    y
end

function ∇lpnorm!{T,P}(x::Array{T}, p::P, gx::Array{T}, gy::Array{T})

end
