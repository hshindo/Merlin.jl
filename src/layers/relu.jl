export relu

function relu{T}(x::Var{T})
    y = Var(x, T, size(x))
    relu!(y.data, x.data)
    y.df = ()
    y
end

function relu(sess::Session, vx::ArratVar{T})
    x, gx = vx.data, vx.grad
    vy = similar(sess, vx)
    y, gy = vy.data, vy.grad
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    function df()
        @inbounds @simd for i = 1:length(x)
            gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
        end
    end
    vy.df = df
    vy
end

function relu{T}(mp::MemoryPool, x::Array{T}, gx::Array{T})
    y = similar(mp, x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end

    function df{T}(gy::Array{T})
        @inbounds @simd for i = 1:length(x)
            gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
        end
    end
    y, df
end

backward!(f::ReLU) = ∇relu!(l.grad, l[1].data, l[1].grad)

function relu!{T}(y::Array{T}, x::Array{T})
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
end

function ∇relu!{T}(gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end
