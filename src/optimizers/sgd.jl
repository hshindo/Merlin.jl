type SGD <: Optimizer
  learnrate::Float64
end

function update!{T}(opt::SGD, param::Array{T}, grad::Array{T})
  axpy!(-T(opt.learnrate), grad, param)
  fill!(grad, T(0.0))
end

function getraw(in::AFArray)
  p = device_ptr(in)
  pp = convert(Ptr{Float32}, p)
  pointer_to_array(pp, jl_size(in))
end

function update!(opt::SGD, v::Variable)
  eltype(v.value) == eltype(v.grad) || throw("type mismatch")
  size(v.value) == size(v.grad) || throw("size mismatch")
  T = eltype(v.value)
  #value = getraw(v.value)
  #grad = getraw(v.grad)
  #axpy!(-T(opt.learnrate), grad, value)
  #unlock_device_ptr(v.value)
  #unlock_device_ptr(v.grad)
  v.value -= T(opt.learnrate) * v.grad
  v.grad = nothing
end
