module Caffe

using ProtoBuf
import ..Merlin
using ..Merlin: Var

include("proto/caffe_pb.jl")

function gksp(p)
  ksize = p.kernel_h > 0 ? (p.kernel_h, p.kernel_w) : (p.kernel_size[1], p.kernel_size[1])
  stride = p.stride_h > 0 ? (p.stride_h, p.stride_w) : (p.stride[1], p.stride[1])
  if p.pad_h > 0
    pad = p.pad_h, p.pad_w
  elseif length(p.pad) > 0
    pad = p.pad[1], p.pad[1]
  else
    pad = 0, 0
  end
  map(Int,ksize), map(Int,stride), map(Int,pad)
end

function conv(layer)
  blobs = layer.blobs
  w = blobs[1].data
  b = length(blobs) > 1 ? blobs[2].data : nothing
  num = blobs[1].num > 0 ? Int(blobs[1].num) : Int(blobs[1].shape.dim[1])
  channels = blobs[1].channels > 0 ? Int(blobs[1].channels) : Int(blobs[1].shape.dim[2])
  param = layer.convolution_param
  ksize, stride, pad = gksp(param)

  param.group == 1 && (w = reshape(w, (ksize[1],ksize[2],channels,num)))
  Merlin.Conv(Var(w), Var(b), stride, pad)
end

function pooling(layer)
  param = layer.pooling_param
  ksize, stride, pad = gksp(param)

  if param.pool == PoolingParameter_PoolMethod.MAX
    mode = "max"
  elseif param.pool == PoolingParameter_PoolMethod.AVE
    mode = "average"
  else
    throw("Unknwon pooling mode.")
  end
  Merlin.Pooling(mode, ksize, stride, pad)
end

function forward(l::LayerParameter, x::Var)
  t = l._type
  t == "Convolution" && return conv(l)(x)
  t == "Pooling" && return pooling(l)(x)
  t == "ReLU" && return Merlin.relu(x)
  #t == types.SOFTMAX_LOSS && return
  x
end

function forward(l::V1LayerParameter, x::Var)
  t = l._type
  types = V1LayerParameter_LayerType
  t == types.CONVOLUTION && return conv(l)(x)
  t == types.POOLING && return pooling(l)(x)
  t == types.RELU && return Merlin.relu(x)
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
