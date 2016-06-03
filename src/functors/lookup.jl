type Lookup; end

lookup(w::Var, x::Var) = init(Lookup(), [w,x])

@compat function (f::Lookup){T}(w::Matrix{T}, x::Matrix{Int})
  y = Array(T, len*size(x,1), size(x,2))
  for i = 1:length(x)
    copy!(y, i*size(w,1), ws[x[i]].value, 1, size(w,1))
  end
  f, y
end

function lookup{T}(w::CuMatrix{T}, x::CuMatrix{Int})

end

function ∇lookup!{T}(w::Matrix{T}, x::Matrix{Int}, gy::Matrix{T})
  for j = 1:size(x,2)
    for i = 1:size(x,1)
      BLAS.axpy!(size(w,1), gy, slice(w,:,j))
    end
  end
end
