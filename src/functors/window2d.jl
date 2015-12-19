const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)

type Window2D <: Functor
  winsize::Tuple{Int,Int}
  stride::Tuple{Int,Int}
  padsize::Tuple{Int,Int}
end

Window2D(winsize, stride) = Window2D(winsize, stride, (0,0))

function apply(fun::Window2D, input::Matrix{Float32})
  w, s, p = fun.winsize, fun.stride, fun.padsize
  colsize = prod(fun.winsize)
  n1 = (size(input, 1) + 2*p[1] - w[1]) ÷ s[1] + 1
  n2 = (size(input, 2) + 2*p[2] - w[2]) ÷ s[2] + 1
  output = Array(Float32, colsize, n1*n2)
  ccall(WINDOW2D_FWD_F32_HANDLE, Void,
    (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
    input, output, size(input, 1), size(input, 2),
    w[1], w[2], s[1], s[2], p[1], p[2])
  output, gy -> diff(fun, input, gy)
end

function diff(fun::Window2D, input::Matrix{Float32}, gradout::Matrix{Float32})
  gradin = zeros(input)
  w, s, p = fun.winsize, fun.stride, fun.padsize
  ccall(WINDOW2D_BWD_F32_HANDLE, Void,
    (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
    gradout, gradin, size(gradin, 1), size(gradin, 2),
    w[1], w[2], s[1], s[2], p[1], p[2])
  gradin
end
