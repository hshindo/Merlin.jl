export Conv
import Base.conv

const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)

im2col_handle(::Type{Float32}) = IM2COL_F32
∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32

"""
    Conv(T::Type, dims::Tuple, [pads], [strides])

N-dimensional convolution function.

* dims: tuple of ``k1, k2, ..., c_{in}, c_{out}``
where ``k1``, ``k2``, ... are kernel size.
``c_{in}`` and ``c_{out}`` are input channel and output channel, respectively.
* [pads=(0,0,...)]: spatial padding size.
* [strides=(1,1,...)]: kernel strides.

```julia
T = Float32
x = Var(rand(T,5,4,3,2))
f = Conv(T, (2,2,3,4), pads=(0,0), strides=(1,1))
y = f(x)
```
```julia
w = zerograd(uniform(T,-0.01,0.01,2,2,3,4))
b = zerograd(zeros(T,4))
f = Conv(w, b, pads=(0,0), strides=(1,1))
y = f(x)
```
"""
type Conv{N}
    w::Var
    b::Var
    pads::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function Conv{N}(T::Type, dims::NTuple{N,Int}; pads=nothing, strides=nothing)
    w = uniform(T, -0.01, 0.01, dims)
    b = zeros(T, dims[end])
    pads == nothing && (pads = ntuple(_ -> 0, N-2))
    strides == nothing && (strides = ntuple(_ -> 1, N-2))
    Conv(zerograd(w), zerograd(b), pads, strides)
end

function (f::Conv)(x::Var)
    setbackend!(f.w, typeof(x.data))
    setbackend!(f.b, typeof(x.data))
    conv(x, f.w, f.b, f.pads, f.strides)
end

function forward{T}(::typeof(conv), x::Array{T}, w::Array{T,4}, b::Vector{T},
    pads::NTuple{2,Int}, strides::NTuple{2,Int})

    ndims(x) == 3 && (x = reshape(x, size(x)..., 1))
    size(x,3) == size(w,3) || throw("Input channel size mismatch.")

    outsize = ntuple(i -> (size(x,i)+2pads[i]-size(w,i)) ÷ strides[i] + 1, 2)
    work = ones(T, prod(outsize)*size(x,4), size(w,1)*size(w,2)*size(w,3))
    ccall(im2col_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        x, work, size(x,1), size(x,2), size(x,3), size(x,4),
        size(w,1), size(w,2), pads[1], pads[2], strides[1], strides[2])

    _w = reshape(w,size(work,2),size(w,4))
    y = work * _w
    broadcast!(.+, y, y, reshape(b,1,length(b)))
    y = reshape(y, outsize..., size(x,4), size(w,4))
    y = permutedims(y, [1,2,4,3])

    function backward!(gy, gx, gw, gb)
        gy = permutedims(gy, [1,2,4,3])
        gy_mat = reshape(gy, size(gy,1)*size(gy,2)*size(gy,3), size(gy,4))
        isvoid(gx) || ∇conv_x!(gy_mat, gx, w, work, pads, strides)
        isvoid(gw) || ∇conv_w!(gy_mat, gw, work)
        isvoid(gb) || BLAS.axpy!(T(1), sum(gy_mat,1), gb)
    end
    y, backward!
end

function forward{T}(::typeof(conv), x::CuArray{T}, w::CuArray{T,4}, b::CuVector{T},
    pads::NTuple{2,Int}, strides::NTuple{2,Int})

    h = handle(x)
    xdesc = CUDNN.TensorDesc(x)
    wdesc = CUDNN.TensorDesc(w)
    ydesc = CUDNN.TensorDesc(y)

    p = cudnnConvolutionFwdAlgo_t[0]
    cudnnGetConvolutionForwardAlgorithm(handle(x), xdesc, wdesc, desc, ydesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionForwardWorkspaceSize(handle(x), xdesc, wdesc, desc, ydesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(worksize))

    cudnnConvolutionForward(handle(x), T[alpha], xdesc, x, wdesc, w, desc,
        algo, workspace, worksize, T[beta], ydesc, y)

    function backward!(gy, gx, gw, gb)
        isvoid(gb) || cudnnConvolutionBackwardBias(h, T[1], dydesc, dy, T[0], dbdesc, gb)
    end
    y, backward!
end

function ∇conv_x!{T}(gy_mat::Matrix{T}, gx::Array{T,4}, w::Array{T,4}, work::Matrix{T}, pads, strides)
    w_mat = reshape(w, size(work,2), size(w,4))
    gwork = similar(work)
    BLAS.gemm!('N', 'T', T(1), gy_mat, w_mat, T(0), gwork)
    ccall(∇im2col_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        gwork, gx, size(gx,1), size(gx,2), size(gx,3), size(gx,4),
        size(w,1), size(w,2), pads[1], pads[2], strides[1], strides[2])
end

function ∇conv_w!{T}(gy_mat::Matrix{T}, gw::Array{T,4}, work::Matrix{T})
    gw_mat = reshape(gw, size(work,2), size(gw,4))
    BLAS.gemm!('T', 'N', T(1), work, gy_mat, T(1), gw_mat)
end

function ∇conv_b!{T}(dy::CuArray{T}, db::CuArray{T})
    dydesc = TensorDesc(dy)
    dbdesc = TensorDesc(db)
    cudnnConvolutionBackwardBias(handle(dy), T[1], dydesc, dy, T[0], dbdesc, db)
end
