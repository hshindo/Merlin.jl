# limited to VGG_ILSVR_19_layers.caffemodel
#=
  `caffe.jl` and `caffe_pb.jl` are supposed to be created from
  `caffe/src/caffe/proto/caffe.proto` whose commit is 823d055
  on 3 Jun 2015 (older or newer commits are not recommended)
  by using `protoc --julia_out` command supplied by protocol
  buffers and ProtoBuf.jl.
=#

include("caffe.jl")

module CaffeFunc

type _param
  name
  value
end

type _layer
  name
  _type
  param
end

function _get_kernel(param)
  if 0 < param.kernel_h
    [param.kernel_h,param.kernel_w]
  else
    fill(param.kernel_size,2)
  end
end

function _get_stride(param)
  if 0 < param.stride_h
    [param.stride_h,param.stride_w]
  else
    fill(param.stride,2)
  end
end

function _get_pad(param)
  if 0 < param.pad_h
    [param.pad_h,param.pad_w]
  else
    fill(param.pad,2)
  end
end

function _get_num(blob)
  if 0 < blob.num
    blob.num
  else
    blob.shape.dim[1]
  end
end

function _get_channels(blob)
  if 0 < blob.channels
    blob.channels
  else
    blob.shape.dim[2]
  end
end

function _convolution(layer)
  x = _layer(layer.name,"Convolution",Array{_param,1}())
  blobs = layer.blobs
  param = layer.convolution_param
  kernel = _param("kernel_hw",_get_kernel(param))
  stride = _param("stride_hw",_get_stride(param))
  pad = _param("pad_hw",_get_pad(param))
  num = _param("num",_get_num(blobs[1]))
  channels = _param("channels",_get_channels(blobs[1]))

  d4 = Int64(num.value)
  d3 = channels.value*param.group
  d1 = d2 = Int(sqrt(length(blobs[1].data)/(d4*d3)))
  w = _param("w",reshape(blobs[1].data,(d1,d2,d3,d4)))   # assume 1 == param.group

  push!(x.param,kernel)
  push!(x.param,stride)
  push!(x.param,pad)
  push!(x.param,num)
  push!(x.param,channels)
  push!(x.param,w)

  if param.bias_term
    bias = _param("bias",blobs[2].data)
    push!(x.param,bias)
  end
  x
end

function _relu(layer)
  x = _layer(layer.name,"ReLU",Array{_param,1}())
# ERROR: UndefRefError: access to undefined reference
#=
  slope = _param("slope",layer.relu_param.negative_slope)
  push!(x.param,slope)
=#
  x
end

function _pooling(layer)
  x = _layer(layer.name,"Pooling",Array{_param,1}())
  param = layer.pooling_param
  kernel = _param("kernel_hw",_get_kernel(param))
  stride = _param("stride_hw",_get_stride(param))
  pad = _param("pad_hw",_get_pad(param))

#  if param.MAX == param.pool   # ERROR: type PoolingParameter has no field MAX
  if 0 == param.pool   # assume 0 means MAX
    pool = _param("pool","MAX")
  else   # assume param.AVE == param.pool
    pool = _param("pool","AVE")
  end

  push!(x.param,kernel)
  push!(x.param,stride)
  push!(x.param,pad)
  push!(x.param,pool)
  x
end

function load(path,print_info)
  f = open(path)
  d = Main.ProtoBuf.readproto(f,Main.caffe.NetParameter())
  layers = Array(_layer,0)
  for layer in d.layers   # assume false == d.layer
    if      4 == layer._type
      push!(layers,_convolution(layer))
    elseif 17 == layer._type
      push!(layers,_pooling(layer))
    elseif 18 == layer._type
      push!(layers,_relu(layer))
    end
  end

  if print_info
    for i = 1:length(layers)
      println("layer[$(i)]:$(layers[i].name) $(layers[i]._type)")
      for j = 1:length(layers[i].param)
        print("  param[$(j)]:$(layers[i].param[j].name) ")
        print("$(typeof(layers[i].param[j].value))")
        if ASCIIString == typeof(layers[i].param[j].value) || 1 == length(layers[i].param[j].value)
          println(" = $(layers[i].param[j].value)")
        elseif 2 == length(layers[i].param[j].value)
          println(" = [$(layers[i].param[j].value[1]),$(layers[i].param[j].value[2])]")
        else
          println("$(size(layers[i].param[j].value))")
        end
      end
    end
  end
  layers
end

end

caffe_func(path,print_info = false) = CaffeFunc.load(path,print_info)
