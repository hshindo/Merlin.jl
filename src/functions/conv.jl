export ConvFun
import Base.conv

const WINDOW2D_FWD_F32 = Libdl.dlsym(library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32 = Libdl.dlsym(library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64 = Libdl.dlsym(library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64 = Libdl.dlsym(library, :window2d_bwd_f64)

"""
    Conv(w, x; [stride, pad])

N-dimensional convolution function.

## Arguments
* w::Var: weight
* x::Var: input
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* pad::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
w = Var(rand(Float32,2,2,3,4))
y = Conv(w, stride=(1,1), pad=(0,0))(x)
```
"""
type Conv{N}
  w::Var
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

"""
    Conv()
"""
function Conv{N}(stride, pad)
end

@compat function (f::Conv{N}){N}(args::Vector{Var})
  @checkargs f args
  w, x = args[1], args[2]
  y = conv(x.value, w.value, f.stride, f.pad)
  df(y::Var) = throw("Not implemented yet.")
  Var(y, df, [w,x])
end

@compat (f::Conv{N}){N}(x::Var) = f([f.w,x])

function conv{T}(x::Array{T,4}, w::Array{T,4}, stride, pad)
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

function conv(w::CuArray, stride, pad, x::CuArray)
  CUDNN.convolution!(x, w, pad, stride)
end

function ∇conv!{T}(w::Array{T,4}, stride::NTuple{2,Int}, pad::NTuple{2,Int}, x::Array{T,4})

end
=#
