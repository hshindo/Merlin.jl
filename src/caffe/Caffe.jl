module Caffe

using ProtoBuf
import ..Merlin
using ..Merlin: Var

include("caffe_pb.jl")

#function __init__()
#end

"""
Load Caffe model.

## 👉 Example
```julia
using Merlin
using Merlin.Caffe

path = "<xxx.caffemodel>"
model = Caffe.load(path)
```
"""
function load(path)
  np = open(path) do io
    readproto(io, NetParameter())
  end

  if length(np.layer) > 0
    layers = np.layer
    dict = Dict(
      "Convolution" => conv,
      "Data" => data,
      "Dropout" => dropout,
      "InnerProduct" => inner_product,
      "Pooling" => pooling,
      "ReLU" => relu,
      "Softmax" => softmax,
      "SoftmaxWithLoss" => softmax_loss,
    )
  else
    layers = np.layers
    t = V1LayerParameter_LayerType
    dict = Dict(
      t.CONVOLUTION => conv,
      t.DATA => data,
      t.DROPOUT => dropout,
      t.INNER_PRODUCT => inner_product,
      t.POOLING => pooling,
      t.RELU => relu,
      t.SOFTMAX => softmax,
      t.SOFTMAX_LOSS => softmax_loss,
    )
  end

  x = Var(:x)
  names = Dict()
  for l in layers
    names[l.name] = l
    f = dict[l._type]
    #x = f(x)
  end
  x
  #Merlin.@graph x
end

function conv(layer)
  blobs = layer.blobs
  w = blobs[1].data
  b = length(blobs) > 1 ? blobs[2].data : nothing
  num = blobs[1].num > 0 ? Int(blobs[1].num) : Int(blobs[1].shape.dim[1])
  channels = blobs[1].channels > 0 ? Int(blobs[1].channels) : Int(blobs[1].shape.dim[2])
  param = layer.convolution_param
  kernel, stride, pad = kernel_size(param), stride_size(param), pad_size(param)

  # group
  #=
  ngroups = param.group
  n_in = channels * ngroups
  n_out = num
  for i = 1:ngroups
    in_slice = slice((i-1) * n_in, i * n_in)
    out_slice = slice((i-1) * n_out, i * n_out)
    w = func.W.data[out_slice, in_slice]
  end
  p.group == 1 && (w = reshape(w, kernel[1], kernel[2], channels, num))
  =#

  Merlin.Conv(Var(w), Var(b), stride, pad)
end

function data(layer)
end

function dropout(layer)
  ratio = Float64(layer.dropout_param.dropout_ratio)
  Merlin.Dropout(ratio)
end

function inner_product(layer)
  param = layer.inner_product_param
  if param.axis != 1
    throw("Axis: $(axis) in inner-product is unsupported.")
  end

  blobs = layer.blobs
  b = blobs[1]
  if blob.height > 0
    height, width = Int(b.height), Int(b.width)
  elseif length(b.shape.dim) == 2
    height, width = Int(b.shape.dim[1]), Int(b.shape.dim[2])
  elseif length(b.shape.dim) == 4
    height, width = Int(b.shape.dim[3]), Int(b.shape.dim[4])
  else
    throw("Dimensions: $(b.shape.dim) in inner-product is unsupported.")
  end

  w = reshape(blobs[1].data, height, width)
  b = param.bias_term ? vec(blobs[2].data) : nothing
  Merlin.Linear(Var(w), Var(b))
end

function pooling(layer)
  param = layer.convolution_param
  kernel, stride, pad = kernel_size(param), stride_size(param), pad_size(param)
  types = PoolingParameter_PoolMethod
  if p.pool == types.MAX
    mode = :max
  elseif p.pool == types.AVE
    mode = :ave
  else
    warn("Mode: $(mode) in Pooling is unsupported. Set to max.")
    mode = :max
  end
  Merlin.Pooling(mode, kernel, stride, pad)
end

function relu(layer)
  slope = Float64(layer.relu_param.negative_slope)
  if slope != 0.0
    warn("Leaky ReLU is unsupported. Slope is set to 0.")
  end
  Merlin.relu
end

function softmax(layer)
  axis = Int(layer.softmax_param.axis)
  Merlin.Softmax(axis)
end

function softmax_loss(layer)
  axis = Int(layer.softmax_param.axis)
  Merlin.SoftmaxCrossEntropy(axis)
end

########## Internal ##########

function kernel_size(param)
  param.kernel_h > 0 && return Int(param.kernel_h), Int(param.kernel_w)
  s = Int(param.kernel_size[1])
  s, s
end

function stride_size(param)
  param.stride_h > 0 && return Int(param.stride_h), Int(param.stride_w)
  length(param.stride) == 0 && return 1, 1
  s = Int(param.stride[1])
  s, s
end

function pad_size(param)
  param.pad_h > 0 && return Int(param.pad_h), Int(param.pad_w)
  length(param.pad) == 0 && return 0, 0
  s = Int(param.pad[1])
  s, s
end

end
