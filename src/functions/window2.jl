export window2d

const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(libmerlin, :window_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(libmerlin, :window_bwd_f32)
const WINDOW2D_FWD_F64_HANDLE = Libdl.dlsym(libmerlin, :window_fwd_f64)
const WINDOW2D_BWD_F64_HANDLE = Libdl.dlsym(libmerlin, :window_bwd_f64)

function window2d(x::Var, w1, w2, s1, s2, p1, p2)
    f = Window2D(w1,w2,s1,s2,p1,p2)
    y, params = window2d(f, x.data)
    df(gy) = hasgrad(x) && ∇window2d!(f, params, x.data, x.grad, gy)
    Var(y, [x], window2d, df)
end

type Window2D
    w1::Int
    w2::Int
    s1::Int
    s2::Int
    p1::Int
    p2::Int

    function Window2D(w1, w2, s1, s2, p1=0, p2=0)
        (s1 > 0 && s2 > 0) || throw("stride must be > 0")
        new(w1, w2, s1, s2, p1, p2)
    end
end

fwd_handle(f::Window2D, ::Type{Float32}) = WINDOW2D_FWD_F32_HANDLE
fwd_handle(f::Window2D, ::Type{Float64}) = WINDOW2D_FWD_F64_HANDLE
bwd_handle(f::Window2D, ::Type{Float32}) = WINDOW2D_BWD_F32_HANDLE
bwd_handle(f::Window2D, ::Type{Float64}) = WINDOW2D_BWD_F64_HANDLE

function make_params(f::Window2D)
    Int32[f.w1, f.w2, f.s1, f.s2, f.p1, f.p2]
end

@compat function (f::Window2D)(args::Vector{Var})
    x = args[1]
    y, params = window2d(f, x.value)
    df(gy) = hasgrad(x) && ∇window2d!(f, params, x.value, x.grad, gy)
    Var(y, df, args)
end

function window2d{T}(f::Window2D, x::Matrix{T})
    w1, w2, s1, s2, p1, p2 = f.w1, f.w2, f.s1, f.s2, f.p1, f.p2
    n1 = (size(x,1) + 2*p1 - w1) ÷ s1 + 1
    n2 = (size(x,2) + 2*p2 - w2) ÷ s2 + 1
    params = Int32[w1, w2, s1, s2, p1, p2]
    y = Array(T, w1*w2, n1*n2)
    ccall(fwd_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Cint, Cint),
    x, params, y, size(x,1), size(x,2))
    y, params
end

function ∇window2d!{T}(f::Window2D, params::Vector{Int32}, x::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
    ccall(bwd_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
    params, gy, gx, size(gx,1), size(gx,2))
end
