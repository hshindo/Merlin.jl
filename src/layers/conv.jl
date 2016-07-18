export Conv
import Base.conv

"""
    Conv(w, [stride, pad])

N-dimensional convolution function.

## Arguments
* w::Var: weight (windowsize, input channel, output channel)
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* padsize::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Data(rand(Float32,5,4,3,2))
f = Conv(Param(rand(Float32,2,2,3,4)), stride=(1,1), padsize=(0,0))
y = f(x)
```
"""
@Var(Conv{N},
winsize::NTuple{N,Int},
stride::NTuple{N,Int},
padsize::NTuple{N,Int})

function conv{T}(w::Array{T}, x::Array{T}, winsize, stride, padsize)
  N = length(winsize)

end

function window{T,N}(out::Conv{N}, x::Array{T})
    h = handle(Conv{N}, T)
    y = Array(T, prod(outsize(out,x)), prod(out.winsize)*size(x,N+1), size(x,N+2))
    xsize = Cint[size(x,i) for i=1:N+1]
    xsize[N+1] *= size(x, N+2)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    x, y, xsize, Cint[out.winsize...], Cint[out.stride...], Cint[out.padsize...])
    y
end
