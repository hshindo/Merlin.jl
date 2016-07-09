import Base: .*, *

.*(x1::Layer, x2::Layer) = ElemTimes(x1, x2, x1.y .* x2.y, nothing)

type ElemTimes <: Layer
  x1
  x2
  y
  gy
end

tails(l::ElemTimes) = [l.x1, l.x2]

function backward!(l::ElemTimes)
  hasgrad(l.x1) && ∇elemtimes!(l.x2.y, l.x1.gy, l.gy)
  hasgrad(l.x2) && ∇elemtimes!(l.x1.y, l.x2.gy, l.gy)
end

function ∇elemtimes!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  if length(gx1) < length(gy)
    @inbounds for k = 1:length(gx1):length(gy)
      @simd for i = 1:length(gx1)
        gx1[i] += gy[k+i-1] * x2[k+i-1]
      end
    end
  else
    broadcast!(.+, gx1, gx1, gy.*x2)
  end
end

*(x1::Layer, x2::Layer) = Times(x1, x2, x1.y * x2.y, nothing)

type Times <: Layer
  x1
  x2
  y
  gy
end

tails(l::Times) = [l.x1, l.x2]

function backward!(l::Times)
  x1, x2, gy = l.x1, l.x2, l.gy
  T = eltype(gy)
  hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.y, T(1), x1.gy)
  hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.y, gy, T(1), x2.gy)
end
