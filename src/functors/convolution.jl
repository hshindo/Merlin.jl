type Convolution1D <: Functor
  w::Vector{Int}
  p::Vector{Int}
  s::Vectir{Int}
end

type Convolution2D <: Functor
end

function forward!(f::Convolution1D)
end

function convlution1d{T,N}(f::Convolution1D, x::Array{T,N})
  y = similar(x)

end

function convlution2d{T,N}(f::Convolution2D, x::Array{T,N})
  y = similar(x)

end

function forward!(f::Convolution2D)
end
