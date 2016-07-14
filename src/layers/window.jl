export window

const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_float)
const WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_double)
const ∇WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_grad_float)
const ∇WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_grad_double)

type Window{N}
  data
  grad
  tails::Vector
  winsize::NTuple{N,Int}
  stride::NTuple{N,Int}
  padsize::NTuple{N,Int}
end

handle(::Type{Window{2}}, ::Type{Float32}) = WINDOW2D_F32
handle(::Type{Window{2}}, ::Type{Float64}) = WINDOW2D_F64
∇handle(::Type{Window{2}}, ::Type{Float32}) = ∇WINDOW2D_F32
∇handle(::Type{Window{2}}, ::Type{Float64}) = ∇WINDOW2D_F64

function window(x::Var, winsize, stride, padsize)
  y = hasdata(x) ? window(x.data,winsize,stride,padsize) : nothing
  Window(y, nothing, [x], winsize, stride, padsize)
end
@compat (f::Window)(x::Var) = window(x, f.winsize, f.stride, f.padsize)

function window{T,N}(x::Array{T}, winsize::NTuple{N,Int}, stride::NTuple{N,Int}, padsize::NTuple{N,Int})
  @assert ndims(x) == N+1
  h = handle(Window{N}, T)
  outdims = outsize(x, winsize, stride, padsize)
  y = Array(T, prod(outdims), prod(winsize)*size(x,N+1))
  xsize = Cint[size(x)...]
  ndims(x)
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    x, y, xsize, Cint[winsize...], Cint[stride...], Cint[padsize...])
  y
end

function outsize(x, winsize, stride, padsize)
  N = length(winsize)
  dims = Array(Int, N)
  for i = 1:N
    dims[i] = (size(x,i) + 2*padsize[i] - winsize[i]) ÷ stride[i] + 1
  end
  dims
end

function backward!(v::Window)
  hasgrad(v[1]) || return
  ∇window!(v.winsize, v.stride, v.padsize, v[1].grad, v.grad)
end

function ∇window!{T,N}(winsize::NTuple{N,Int}, stride, padsize, gx::Array{T}, gy::Array{T})
  h = handle(Window{N}, T)
  xsize = Cint[size(gx)...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    gx, gy, xsize, Cint[winsize...], Cint[stride...], Cint[padsize...])
end
