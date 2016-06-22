module Caffe

using ProtoBuf
import ..Merlin
using ..Merlin: Var

include("proto/caffe_pb.jl")

function kernel_size(param)
  param.kernel_h > 0 && return Int(p.kernel_h), Int(p.kernel_w)
  s = Int(p.kernel_size[1])
  s, s
end

function stride_size(param)
  p.stride_h > 0 && return Int(p.stride_h), Int(p.stride_w)
  length(p.stride) == 0 && return 1, 1
  @assert length(p.stride) == 2
  s = Int(p.stride[1])
  s, s
end

function pad_size(param)
  p.pad_h > 0 && return Int(p.pad_h), Int(p.pad_w)
  length(p.pad) == 0 && return 0, 0
  @assert length(p.pad) == 2
  s = Int(p.pad[1])
  s, s
end

function conv(layer)
  blobs = layer.blobs
  w = blobs[1].data
  b = length(blobs) > 1 ? blobs[2].data : nothing
  num = blobs[1].num > 0 ? Int(blobs[1].num) : Int(blobs[1].shape.dim[1])
  channels = blobs[1].channels > 0 ? Int(blobs[1].channels) : Int(blobs[1].shape.dim[2])
  p = layer.convolution_param
  kernel, stride, pad = kernel_size(p), stride_size(p), pad_size(p)
  p.group == 1 && (w = reshape(w, ksize[1], ksize[2], channels, num))
  Merlin.Conv(Var(w), Var(b), stride, pad)
end

function inner_product(layer)
  w = layer.blobs[1].data
  b = layer.blobs[2].data
  blob = layer.blobs[1]
  if blob.height > 0
    height = blob.height
    width = blob.width
  elseif length(blob.shape.dim) == 2
    height = blob.shape.dim[1]
    width = blob.shape.dim[2]
  elseif length(blob.shape.dim) == 4
    height = blob.shape.dim[3]
    width = blob.shape.dim[4]
  else
    height = nothing
    width = nothing
  end

  w = reshape(w, height, width)
  f = Linear(Var(w), Var(b))

  axis = layer.inner_product_param.axis
  _inner_product(
    w,
    b,
    height,
    width,
    axis
  )
end

function pooling(layer)
  p = layer.pooling_param
  kernel, stride, pad = kernel_size(p), stride_size(p), pad_size(p)
  types = PoolingParameter_PoolMethod
  if p.pool == types.MAX
    mode = :max
  elseif p.pool == types.AVE
    mode = :ave
  else
    warn("Unsupported pooling mode: $(mode). Set the mode :max")
    mode = :max
  end
  Merlin.Pooling(mode, kernel, stride, pad)
end

function forward(l::LayerParameter, x::Var)
  t = l._type
  t == "Convolution" && return conv(l)(x)
  t == "Pooling" && return pooling(l)(x)
  t == "ReLU" && return Merlin.relu(x)
  t == "Softmax" && return Merlin.softmax(x)
  x
end

function forward(l::V1LayerParameter, x::Var)
  t = l._type
  types = V1LayerParameter_LayerType
  t == types.CONVOLUTION && return conv(l)(x)
  if t == types.DROPOUT
    ratio = Float64(layer.dropout_param.dropout_ratio)
    return Merlin.dropout(x, ratio)
  end
  t == types.POOLING && return pooling(l)(x)
  t == types.RELU && return Merlin.relu(x)
  t == types.SOFTMAX && return Merlin.softmax(x)
  x
end

"""
Load Caffe model.
"""
function load(path)
  np = open(path) do io
    readproto(io, NetParameter())
  end
  x = Var(:x)
  layers = length(np.layer) > 0 ? np.layer : np.layers
  for l in layers
    x = forward(l, x)
  end

  Merlin.@graph x
end

end
