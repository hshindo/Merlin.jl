module Caffe

using ProtoBuf
import ..Merlin
import ..Merlin.Var

include("proto/caffe_pb.jl")

function gksp(p)
  ksize = p.kernel_h > 0 ? (p.kernel_h, p.kernel_w) : (p.kernel_size, p.kernel_size)
  stride = p.stride_h > 0 ? (p.stride_h, p.stride_w) : (p.stride, p.stride)
  pad = p.pad_h > 0 ? (p.pad_h, p.pad_w) : (p.pad, p.pad)
  map(Int,ksize), map(Int,stride), map(Int,pad)
end

function conv(layer)
  blobs = layer.blobs
  w = blobs[1].data
  b = length(blobs) > 1 ? blobs[2].data : nothing
  num = blobs[1].num > 0 ? blobs[1].num : blobs[1].shape.dim[1]
  channels = blobs[1].channels > 0 ? blobs[1].channels : blobs[1].shape.dim[2]
  param = layer.convolution_param
  ksize, stride, pad = gksp(param)

  if param.group == 1
    w = reshape(w, (ksize[1],ksize[2],Int64(channels),Int64(num)))
  end
  Merlin.Conv(Merlin.Var(w), Merlin.Var(b), stride, pad)
end

function pooling(layer)
  param = layer.pooling_param
  ksize,stride,pad = gksp(param)

  if param.pool == __enum_PoolingParameter_PoolMethod().MAX
    mode = "max"
  elseif param.pool == __enum_PoolingParameter_PoolMethod().AVE
    mode = "average"
  else
    mode = nothing
  end
  Merlin.Pooling{length(pad)}(mode,ksize,stride,pad)
end

type _data
  backend
  batch_size
  crop_size
  force_encoded_color
  mean_file
  mirror
  scale
  source
end

function data(layer)
  param = layer.data_param
  _data(
    param.backend,
    param.batch_size,
    param.crop_size,
    param.force_encoded_color,
    param.mean_file,
    param.mirror,
    param.scale,
    param.source
  )
end

type _softmax_loss
  axis
end

function softmax_loss(layer)
  if isdefined(layer,:softmax_param)
    axis = layer.softmax_param.axis
  else
    axis= __val_SoftmaxParameter[:axis]
  end
  _softmax_loss(axis)
end

type _relu end

"""
Load Caffe model.
"""
function load(path)
  np = open(path) do io
    readproto(io, NetParameter())
  end
  ltype = __enum_V1LayerParameter_LayerType()
  x = Merlin.Var(:x)
  d=Any[]
  for l in (length(np.layer) > 0 ? np.layer : np.layers)
    f = nothing
    if l._type == ltype.CONVOLUTION || l._type == "Convolution"
      f = conv(l)
      #x = f(x)
    elseif l._type == ltype.POOLING || l._type == "Pooling"
      f = pooling(l)
      x = f(x)
    elseif l._type == ltype.RELU || l._type == "ReLU"
      f = _relu()
      x = Merlin.relu(x)
    elseif l._type == ltype.DATA || l._type == "Data"
      f = data(l)
      #x = f(x)
    elseif l._type == ltype.SOFTMAX_LOSS || l._type == "SoftmaxWithLoss"
      f = softmax_loss(l)
      #x = f(x)
    end
    push!(d,(l.name,typeof(f),l._type))
  end
  x,d
end

end
