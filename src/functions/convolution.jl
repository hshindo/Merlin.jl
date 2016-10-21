const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)

im2col_handle(::Type{Float32}) = IM2COL_F32
∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32

function im2col!{T}(x::Ptr{T}, y::Ptr{T}, xsize, filtersize, padding, stride)
    h = im2col_handle(T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        x, y, xsize..., filtersize..., padding..., stride...)
    y
end

type Convolution{N} <: Functor
    w::Var
    padding::NTuple{N,Int}
    stride::NTuple{N,Int}
end

function Convolution(T::Type, filtersize, channelsize, padding, stride)
    N = length(filter)
    w = rand(T, filtersize..., channelsize...)
    w .*= 0.002
    w .-= 0.001
    Convolution(Var(w), padding, stride)
end

@graph function (f::Convolution)(x::Var)
    y = convolution(x.data, f.w.data, f.padding, f.stride)

    #y, work = conv(f.w.data, x.data, f.filterdims, f.stride, f.paddims)
    function df(gy)
        ∇conv!(f.w.data, f.w.grad, x.data, x.grad, work, y, gy,
        f.filterdims, f.stride, f.paddims)
    end
    Var(y, [x], convolution, df)
end

function convolution{T}(x::Array{T,4}, w::Array{T,4}, padding::Tuple{Int,2}, stride::Tuple{Int,2})
    outdims = Int[(size(x,i)+2padding[i]-size(w,i)) ÷ stride[i] + 1 for i=1:2]
    work = similar(x, outdims[1]*outdims[2], size(w,1)*size(w,2)*size(x,3), size(x,4))
    h = im2col_handle(T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        x, y, size(x,1), size(x,2), size(x,3)*size(x,4),
        size(w,1), size(w,2), padding..., stride...)

    [view(As[i],:,:,i) for i=1:size(As,3)]

    N = length(padding)
    outdims = Int[(size(x,i)+2padding[i]-size(w,i)) ÷ stride[i] + 1 for i=1:N]
    work = similar(x, prod(outdims), size(w,1)*size(w,2)*size(x,N+1), size(x,N+2)) # assuming ndims(x) == 4
    im2col!(x, work, windims, stride, paddims)

    w = reshape(w, size(work,2), size(w,N+2))
    y = gemm(work, w)
    y = reshape(y, outdims..., size(y,2), size(y,3))
    y
end

convolution(x::CuArray, w::CuArray, padding, stride) = CUDNN.convolution(x, w, padding, stride)
