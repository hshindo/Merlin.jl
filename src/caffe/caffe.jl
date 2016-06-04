module Caffe

using ProtoBuf

include("proto/caffe_pb.jl")

function conv(layer)
  blobs = layer.blobs
  w = blobs[1].data
  b = length(blobs) > 1 ? blobs[2].data : nothing
  w, b

  #=
  p = layer.convolution_param
  kernel = p.kernel_h > 0 ? [p.kernel_h, p.kernel_w] : fill(p.kernel_size, 2)
  stride = p.stride_h > 0 ? [p.stride_h, p.stride_w] : fill(p.stride, 2)
  pad = p.pad_h > 0 ? [p.pad_h, p.pad_w] : fill(p.pad, 2)

  b = layer.blobs
  num = blob.num > 0 ? blob.num : blob.shape.dim[1]
  channels = blob.channels > 0 ? blob.channels : blob.shape.dim[2]
  =#
end

function load(path)
  np = open(path) do io
    readproto(io, NetParameter())
  end
  dict = Dict()
  for l in np.layers
    dict[l.name] = l
  end
  dict
end

end
