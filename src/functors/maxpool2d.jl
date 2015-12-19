const MAXPOOL2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :maxpool2d_fwd_f32)
const MAXPOOL2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :maxpool2d_bwd_f32)

type MaxPool2D <: Functor
  winsize::Tuple{Int,Int}
  stride::Tuple{Int,Int}
  padsize::Tuple{Int,Int}
end

MaxPool2D(winsize, stride) = MaxPool2D(winsize, stride, (0,0))

function apply{T}(fun::MaxPool2D, input::Matrix{T})
  w, s, p = fun.winsize, fun.stride, fun.padsize
  w[1] < 0 && (w = (size(input, 1), w[2]))
  w[2] < 0 && (w = (w[1], size(input, 2)))
  d1 = (size(input, 1) + 2*p[1] - w[1]) ÷ s[1] + 1
  d2 = (size(input, 2) + 2*p[2] - w[2]) ÷ s[2] + 1
  maxidxs = Array(Int, d1, d2)
  output = Array(T, d1, d2)
  ccall(MAXPOOL2D_FWD_F32_HANDLE, Void,
    (Ptr{Float32}, Ptr{Cssize_t}, Ptr{Float32}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
    input, maxidxs, output, size(input, 1), size(input, 2),
    w[1], w[2], s[1], s[2], p[1], p[2])
  output, gy -> diff(fun, input, maxidxs, gy)
end

function diff{T}(fun::MaxPool2D, input::Matrix{T}, maxidxs::Matrix{Int}, gradout::Matrix{T})
  gradin = zeros(input)
  ccall(MAXPOOL2D_BWD_F32_HANDLE, Void,
    (Ptr{Cssize_t}, Ptr{Float32}, Ptr{Float32}, Cint),
    maxidxs, gradout, gradin, length(gradout))
  gradin
end
