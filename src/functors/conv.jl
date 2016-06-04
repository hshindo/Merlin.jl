export Conv
import Base.conv

const WINDOW2D_FWD_F32 = Libdl.dlsym(library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32 = Libdl.dlsym(library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64 = Libdl.dlsym(library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64 = Libdl.dlsym(library, :window2d_bwd_f64)

"""
    conv(w, x; [b::Var, stride, pad])

N-dimensional convolution function.

## Arguments
* w::Var: weight
* x::Var: input
* b::Var: bias
* stride::NTuple{N,Int}: stride size
* pad::NTuple{N,Int}: padding size

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
w = Var(rand(Float32,2,2,3,4))
y = conv(w, x, stride=(1,1), pad=(0,0))
```
"""
function conv(w::Var, x::Var; b::Var=Var(), stride=(), pad=())
  N = ndims(w.value) - 2
  isempty(stride) && (stride = ntuple(_ -> 1, N))
  isempty(pad) && (pad = ntuple(_ -> 0, N))

  y = conv(w.value, x.value, stride, pad)
  f(y::Var) = ∇conv!
  Var(y, nothing, f, [w,x,b])
end

#=

handle(::Type{Conv{2}}, ::Type{Float32}) = WINDOW2D_FWD_F32, WINDOW2D_BWD_F32
handle(::Type{Conv{2}}, ::Type{Float64}) = WINDOW2D_FWD_F64, WINDOW2D_BWD_F64


forward(f::Conv, args) = f, conv(f, args[1], args[2])

backward!(f::Conv, y::Var) = ∇conv!

function outsize{N}(f::Conv{N})
  dims = Array(Int, N+2)
  for i = 1:N
    dims[i] = (size(x,i) + 2*f.pad[i] - size(w,i)) ÷ f.stride[i] + 1
  end
  #dims[N+1] =
end

function conv{T}(w::Array{T,4}, x::Array{T,4}, stride, pad)
  stride, pad = f.stride, f.pad
  h = handle(Conv{2}, T)[1]
  xsize = Cint[size(x,1), size(x,2), size(x,3)*size(x,4)]
  params = Cint[size(w,1), size(w,2), stride..., pad...]

  dims = Array(Int, 2)
  for i = 1:length(dims)
    dims[i] = (size(x,i) + 2*pad[i] - size(w,i)) ÷ stride[i] + 1
  end
  work = similar(x, prod(dims), size(w,1)*size(w,2)*size(w,3), size(x,4))
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, work, xsize, params)

  w = reshape(w, size(w,1)*size(w,2)*size(w,3), size(w,4))
  y = similar(x, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,4)
    BLAS.gemm!('N', 'N', T(1), slice(work,:,:,i), w, T(0), slice(y,:,:,i))
  end
  y = reshape(y, dims[1], dims[2], size(y,2), size(y,3))
  y
end

function conv(w::CuArray, stride, pad, x::CuArray)
  CUDNN.convolution!(x, w, pad, stride)
end

function ∇conv!{T}(w::Array{T,4}, stride::NTuple{2,Int}, pad::NTuple{2,Int}, x::Array{T,4})

end
=#
