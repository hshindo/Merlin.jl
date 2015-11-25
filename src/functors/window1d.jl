const WINDOW1D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window1d_fwd_f32)
const WINDOW1D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window1d_bwd_f32)

type Window1D <: Functor
  winsize::Int
  stride::Int
  padsize::Int

  function Window1D(winsize, stride, padsize)
    (winsize > 0) || error("size <= 0")
    (stride > 0) || error("stride <= 0")
    (padsize >= 0) || error("stride < 0")
    new(winsize, stride, padsize)
  end
end

function outsize(fun::Window1D, input)
  n = trunc(Int, (length(input) + 2(fun.padsize) - fun.winsize) / fun.stride) + 1
  n > 0 || error("invalid window1d setting")
  fun.winsize, n
end

function apply(fun::Window1D, var::Variable)
  input = var.data
  output = Array(Float32, outsize(fun, input))
  ccall(WINDOW1D_FWD_F32_HANDLE, Void,
    (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cint),
    input, output, length(input), fun.winsize, fun.stride, fun.padsize)
  Variable(output)
end

function apply(fun::Window1D, input::Array{Float32})
  output = Array(Float32, outsize(fun, input))
  ccall(WINDOW1D_FWD_F32_HANDLE, Void,
    (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cint),
    input, output, length(input), fun.winsize, fun.stride, fun.padsize)
  (output,)
end

function diff(fun::Window1D, input::Array{Float32}, gradout::Array{Float32})
  gradin = similar(input)
  ccall(WINDOW1D_BWD_F32_HANDLE, Void,
    (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cint),
    input, gradout, gradin, length(input), fun.winsize, fun.stride, fun.padsize)
  gradin
end

##### old #####
function apply_jl!{T}(fun::Window1D, input::Array{T}, output)
  inidx = -fun.padsize + 1
  outidx = 1
  while outidx <= length(output)
    for i = inidx:(inidx + fun.winsize - 1)
      output[outidx] = (i >= 1 && i <= length(input)) ? input[i] : T(0.0)
      outidx += 1
    end
    inidx += fun.stride
  end
end

function diff_jl!{T}(fun::Window1D, input::Array{T}, gradout, gradin)
  fill!(gradin, T(0.0))
  inidx = -fun.padsize + 1
  outidx = 1
  while outidx <= length(gradout)
    for i = inidx:(inidx + fun.winsize - 1)
      if i >= 1 && i <= length(input)
        gradin[i] += gradout[outidx]
      end
      outidx += 1
    end
    inidx += fun.stride
  end
end
