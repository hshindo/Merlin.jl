using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type __enum_Phase <: ProtoEnum
    TRAIN::Int32
    TEST::Int32
    __enum_Phase() = new(0,1)
end #type __enum_Phase
const Phase = __enum_Phase()

type BlobShape
    dim::Array{Int64,1}
    BlobShape(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type BlobShape
const __pack_BlobShape = Symbol[:dim]
meta(t::Type{BlobShape}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_BlobShape, ProtoBuf.DEF_WTYPES)
hash(v::BlobShape) = ProtoBuf.protohash(v)
isequal(v1::BlobShape, v2::BlobShape) = ProtoBuf.protoisequal(v1, v2)
==(v1::BlobShape, v2::BlobShape) = ProtoBuf.protoeq(v1, v2)

type BlobProto
    shape::BlobShape
    data::Array{Float32,1}
    diff::Array{Float32,1}
    double_data::Array{Float64,1}
    double_diff::Array{Float64,1}
    num::Int32
    channels::Int32
    height::Int32
    width::Int32
    BlobProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type BlobProto
const __val_BlobProto = @compat Dict(:num => 0, :channels => 0, :height => 0, :width => 0)
const __fnum_BlobProto = Int[7,5,6,8,9,1,2,3,4]
const __pack_BlobProto = Symbol[:data,:diff,:double_data,:double_diff]
meta(t::Type{BlobProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_BlobProto, __val_BlobProto, true, __pack_BlobProto, ProtoBuf.DEF_WTYPES)
hash(v::BlobProto) = ProtoBuf.protohash(v)
isequal(v1::BlobProto, v2::BlobProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::BlobProto, v2::BlobProto) = ProtoBuf.protoeq(v1, v2)

type BlobProtoVector
    blobs::Array{BlobProto,1}
    BlobProtoVector(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type BlobProtoVector
hash(v::BlobProtoVector) = ProtoBuf.protohash(v)
isequal(v1::BlobProtoVector, v2::BlobProtoVector) = ProtoBuf.protoisequal(v1, v2)
==(v1::BlobProtoVector, v2::BlobProtoVector) = ProtoBuf.protoeq(v1, v2)

type Datum
    channels::Int32
    height::Int32
    width::Int32
    data::Array{UInt8,1}
    label::Int32
    float_data::Array{Float32,1}
    encoded::Bool
    Datum(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type Datum
const __val_Datum = @compat Dict(:encoded => false)
meta(t::Type{Datum}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_Datum, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::Datum) = ProtoBuf.protohash(v)
isequal(v1::Datum, v2::Datum) = ProtoBuf.protoisequal(v1, v2)
==(v1::Datum, v2::Datum) = ProtoBuf.protoeq(v1, v2)

type __enum_FillerParameter_VarianceNorm <: ProtoEnum
    FAN_IN::Int32
    FAN_OUT::Int32
    AVERAGE::Int32
    __enum_FillerParameter_VarianceNorm() = new(0,1,2)
end #type __enum_FillerParameter_VarianceNorm
const FillerParameter_VarianceNorm = __enum_FillerParameter_VarianceNorm()

type FillerParameter
    _type::AbstractString
    value::Float32
    min::Float32
    max::Float32
    mean::Float32
    std::Float32
    sparse::Int32
    variance_norm::Int32
    FillerParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type FillerParameter
const __val_FillerParameter = @compat Dict(:_type => "constant", :value => 0, :min => 0, :max => 1, :mean => 0, :std => 1, :sparse => -1, :variance_norm => FillerParameter_VarianceNorm.FAN_IN)
meta(t::Type{FillerParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_FillerParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::FillerParameter) = ProtoBuf.protohash(v)
isequal(v1::FillerParameter, v2::FillerParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::FillerParameter, v2::FillerParameter) = ProtoBuf.protoeq(v1, v2)

type SolverState
    iter::Int32
    learned_net::AbstractString
    history::Array{BlobProto,1}
    current_step::Int32
    SolverState(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SolverState
const __val_SolverState = @compat Dict(:current_step => 0)
meta(t::Type{SolverState}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_SolverState, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::SolverState) = ProtoBuf.protohash(v)
isequal(v1::SolverState, v2::SolverState) = ProtoBuf.protoisequal(v1, v2)
==(v1::SolverState, v2::SolverState) = ProtoBuf.protoeq(v1, v2)

type NetState
    phase::Int32
    level::Int32
    stage::Array{AbstractString,1}
    NetState(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NetState
const __val_NetState = @compat Dict(:phase => Phase.TEST, :level => 0)
meta(t::Type{NetState}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_NetState, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::NetState) = ProtoBuf.protohash(v)
isequal(v1::NetState, v2::NetState) = ProtoBuf.protoisequal(v1, v2)
==(v1::NetState, v2::NetState) = ProtoBuf.protoeq(v1, v2)

type NetStateRule
    phase::Int32
    min_level::Int32
    max_level::Int32
    stage::Array{AbstractString,1}
    not_stage::Array{AbstractString,1}
    NetStateRule(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NetStateRule
hash(v::NetStateRule) = ProtoBuf.protohash(v)
isequal(v1::NetStateRule, v2::NetStateRule) = ProtoBuf.protoisequal(v1, v2)
==(v1::NetStateRule, v2::NetStateRule) = ProtoBuf.protoeq(v1, v2)

type __enum_ParamSpec_DimCheckMode <: ProtoEnum
    STRICT::Int32
    PERMISSIVE::Int32
    __enum_ParamSpec_DimCheckMode() = new(0,1)
end #type __enum_ParamSpec_DimCheckMode
const ParamSpec_DimCheckMode = __enum_ParamSpec_DimCheckMode()

type ParamSpec
    name::AbstractString
    share_mode::Int32
    lr_mult::Float32
    decay_mult::Float32
    ParamSpec(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ParamSpec
const __val_ParamSpec = @compat Dict(:lr_mult => 1, :decay_mult => 1)
meta(t::Type{ParamSpec}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ParamSpec, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ParamSpec) = ProtoBuf.protohash(v)
isequal(v1::ParamSpec, v2::ParamSpec) = ProtoBuf.protoisequal(v1, v2)
==(v1::ParamSpec, v2::ParamSpec) = ProtoBuf.protoeq(v1, v2)

type TransformationParameter
    scale::Float32
    mirror::Bool
    crop_size::UInt32
    mean_file::AbstractString
    mean_value::Array{Float32,1}
    force_color::Bool
    force_gray::Bool
    TransformationParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TransformationParameter
const __val_TransformationParameter = @compat Dict(:scale => 1, :mirror => false, :crop_size => 0, :force_color => false, :force_gray => false)
meta(t::Type{TransformationParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_TransformationParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::TransformationParameter) = ProtoBuf.protohash(v)
isequal(v1::TransformationParameter, v2::TransformationParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::TransformationParameter, v2::TransformationParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_LossParameter_NormalizationMode <: ProtoEnum
    FULL::Int32
    VALID::Int32
    BATCH_SIZE::Int32
    NONE::Int32
    __enum_LossParameter_NormalizationMode() = new(0,1,2,3)
end #type __enum_LossParameter_NormalizationMode
const LossParameter_NormalizationMode = __enum_LossParameter_NormalizationMode()

type LossParameter
    ignore_label::Int32
    normalization::Int32
    normalize::Bool
    LossParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type LossParameter
const __val_LossParameter = @compat Dict(:normalization => LossParameter_NormalizationMode.VALID)
const __fnum_LossParameter = Int[1,3,2]
meta(t::Type{LossParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_LossParameter, __val_LossParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::LossParameter) = ProtoBuf.protohash(v)
isequal(v1::LossParameter, v2::LossParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::LossParameter, v2::LossParameter) = ProtoBuf.protoeq(v1, v2)

type AccuracyParameter
    top_k::UInt32
    axis::Int32
    ignore_label::Int32
    AccuracyParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type AccuracyParameter
const __val_AccuracyParameter = @compat Dict(:top_k => 1, :axis => 1)
meta(t::Type{AccuracyParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_AccuracyParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::AccuracyParameter) = ProtoBuf.protohash(v)
isequal(v1::AccuracyParameter, v2::AccuracyParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::AccuracyParameter, v2::AccuracyParameter) = ProtoBuf.protoeq(v1, v2)

type ArgMaxParameter
    out_max_val::Bool
    top_k::UInt32
    axis::Int32
    ArgMaxParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ArgMaxParameter
const __val_ArgMaxParameter = @compat Dict(:out_max_val => false, :top_k => 1)
meta(t::Type{ArgMaxParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ArgMaxParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ArgMaxParameter) = ProtoBuf.protohash(v)
isequal(v1::ArgMaxParameter, v2::ArgMaxParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ArgMaxParameter, v2::ArgMaxParameter) = ProtoBuf.protoeq(v1, v2)

type ConcatParameter
    axis::Int32
    concat_dim::UInt32
    ConcatParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ConcatParameter
const __val_ConcatParameter = @compat Dict(:axis => 1, :concat_dim => 1)
const __fnum_ConcatParameter = Int[2,1]
meta(t::Type{ConcatParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConcatParameter, __val_ConcatParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ConcatParameter) = ProtoBuf.protohash(v)
isequal(v1::ConcatParameter, v2::ConcatParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConcatParameter, v2::ConcatParameter) = ProtoBuf.protoeq(v1, v2)

type BatchNormParameter
    use_global_stats::Bool
    moving_average_fraction::Float32
    eps::Float32
    BatchNormParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type BatchNormParameter
const __val_BatchNormParameter = @compat Dict(:moving_average_fraction => 0.999, :eps => 1e-05)
meta(t::Type{BatchNormParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_BatchNormParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::BatchNormParameter) = ProtoBuf.protohash(v)
isequal(v1::BatchNormParameter, v2::BatchNormParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::BatchNormParameter, v2::BatchNormParameter) = ProtoBuf.protoeq(v1, v2)

type BiasParameter
    axis::Int32
    num_axes::Int32
    filler::FillerParameter
    BiasParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type BiasParameter
const __val_BiasParameter = @compat Dict(:axis => 1, :num_axes => 1)
meta(t::Type{BiasParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_BiasParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::BiasParameter) = ProtoBuf.protohash(v)
isequal(v1::BiasParameter, v2::BiasParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::BiasParameter, v2::BiasParameter) = ProtoBuf.protoeq(v1, v2)

type ContrastiveLossParameter
    margin::Float32
    legacy_version::Bool
    ContrastiveLossParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ContrastiveLossParameter
const __val_ContrastiveLossParameter = @compat Dict(:margin => 1, :legacy_version => false)
meta(t::Type{ContrastiveLossParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ContrastiveLossParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ContrastiveLossParameter) = ProtoBuf.protohash(v)
isequal(v1::ContrastiveLossParameter, v2::ContrastiveLossParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ContrastiveLossParameter, v2::ContrastiveLossParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_ConvolutionParameter_Engine <: ProtoEnum
    DEFAULT::Int32
    CAFFE::Int32
    CUDNN::Int32
    __enum_ConvolutionParameter_Engine() = new(0,1,2)
end #type __enum_ConvolutionParameter_Engine
const ConvolutionParameter_Engine = __enum_ConvolutionParameter_Engine()

type ConvolutionParameter
    num_output::UInt32
    bias_term::Bool
    pad::Array{UInt32,1}
    kernel_size::Array{UInt32,1}
    stride::Array{UInt32,1}
    dilation::Array{UInt32,1}
    pad_h::UInt32
    pad_w::UInt32
    kernel_h::UInt32
    kernel_w::UInt32
    stride_h::UInt32
    stride_w::UInt32
    group::UInt32
    weight_filler::FillerParameter
    bias_filler::FillerParameter
    engine::Int32
    axis::Int32
    force_nd_im2col::Bool
    ConvolutionParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ConvolutionParameter
const __val_ConvolutionParameter = @compat Dict(:bias_term => true, :pad_h => 0, :pad_w => 0, :group => 1, :engine => ConvolutionParameter_Engine.DEFAULT, :axis => 1, :force_nd_im2col => false)
const __fnum_ConvolutionParameter = Int[1,2,3,4,6,18,9,10,11,12,13,14,5,7,8,15,16,17]
meta(t::Type{ConvolutionParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConvolutionParameter, __val_ConvolutionParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ConvolutionParameter) = ProtoBuf.protohash(v)
isequal(v1::ConvolutionParameter, v2::ConvolutionParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConvolutionParameter, v2::ConvolutionParameter) = ProtoBuf.protoeq(v1, v2)

type CropParameter
    axis::Int32
    offset::Array{UInt32,1}
    CropParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CropParameter
const __val_CropParameter = @compat Dict(:axis => 2)
meta(t::Type{CropParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_CropParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::CropParameter) = ProtoBuf.protohash(v)
isequal(v1::CropParameter, v2::CropParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::CropParameter, v2::CropParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_DataParameter_DB <: ProtoEnum
    LEVELDB::Int32
    LMDB::Int32
    __enum_DataParameter_DB() = new(0,1)
end #type __enum_DataParameter_DB
const DataParameter_DB = __enum_DataParameter_DB()

type DataParameter
    source::AbstractString
    batch_size::UInt32
    rand_skip::UInt32
    backend::Int32
    scale::Float32
    mean_file::AbstractString
    crop_size::UInt32
    mirror::Bool
    force_encoded_color::Bool
    prefetch::UInt32
    DataParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type DataParameter
const __val_DataParameter = @compat Dict(:rand_skip => 0, :backend => DataParameter_DB.LEVELDB, :scale => 1, :crop_size => 0, :mirror => false, :force_encoded_color => false, :prefetch => 4)
const __fnum_DataParameter = Int[1,4,7,8,2,3,5,6,9,10]
meta(t::Type{DataParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DataParameter, __val_DataParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::DataParameter) = ProtoBuf.protohash(v)
isequal(v1::DataParameter, v2::DataParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::DataParameter, v2::DataParameter) = ProtoBuf.protoeq(v1, v2)

type DropoutParameter
    dropout_ratio::Float32
    DropoutParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type DropoutParameter
const __val_DropoutParameter = @compat Dict(:dropout_ratio => 0.5)
meta(t::Type{DropoutParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_DropoutParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::DropoutParameter) = ProtoBuf.protohash(v)
isequal(v1::DropoutParameter, v2::DropoutParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::DropoutParameter, v2::DropoutParameter) = ProtoBuf.protoeq(v1, v2)

type DummyDataParameter
    data_filler::Array{FillerParameter,1}
    shape::Array{BlobShape,1}
    num::Array{UInt32,1}
    channels::Array{UInt32,1}
    height::Array{UInt32,1}
    width::Array{UInt32,1}
    DummyDataParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type DummyDataParameter
const __fnum_DummyDataParameter = Int[1,6,2,3,4,5]
meta(t::Type{DummyDataParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DummyDataParameter, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::DummyDataParameter) = ProtoBuf.protohash(v)
isequal(v1::DummyDataParameter, v2::DummyDataParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::DummyDataParameter, v2::DummyDataParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_EltwiseParameter_EltwiseOp <: ProtoEnum
    PROD::Int32
    SUM::Int32
    MAX::Int32
    __enum_EltwiseParameter_EltwiseOp() = new(0,1,2)
end #type __enum_EltwiseParameter_EltwiseOp
const EltwiseParameter_EltwiseOp = __enum_EltwiseParameter_EltwiseOp()

type EltwiseParameter
    operation::Int32
    coeff::Array{Float32,1}
    stable_prod_grad::Bool
    EltwiseParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type EltwiseParameter
const __val_EltwiseParameter = @compat Dict(:operation => EltwiseParameter_EltwiseOp.SUM, :stable_prod_grad => true)
meta(t::Type{EltwiseParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_EltwiseParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::EltwiseParameter) = ProtoBuf.protohash(v)
isequal(v1::EltwiseParameter, v2::EltwiseParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::EltwiseParameter, v2::EltwiseParameter) = ProtoBuf.protoeq(v1, v2)

type ELUParameter
    alpha::Float32
    ELUParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ELUParameter
const __val_ELUParameter = @compat Dict(:alpha => 1)
meta(t::Type{ELUParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ELUParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ELUParameter) = ProtoBuf.protohash(v)
isequal(v1::ELUParameter, v2::ELUParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ELUParameter, v2::ELUParameter) = ProtoBuf.protoeq(v1, v2)

type EmbedParameter
    num_output::UInt32
    input_dim::UInt32
    bias_term::Bool
    weight_filler::FillerParameter
    bias_filler::FillerParameter
    EmbedParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type EmbedParameter
const __val_EmbedParameter = @compat Dict(:bias_term => true)
meta(t::Type{EmbedParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_EmbedParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::EmbedParameter) = ProtoBuf.protohash(v)
isequal(v1::EmbedParameter, v2::EmbedParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::EmbedParameter, v2::EmbedParameter) = ProtoBuf.protoeq(v1, v2)

type ExpParameter
    base::Float32
    scale::Float32
    shift::Float32
    ExpParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ExpParameter
const __val_ExpParameter = @compat Dict(:base => -1, :scale => 1, :shift => 0)
meta(t::Type{ExpParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ExpParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ExpParameter) = ProtoBuf.protohash(v)
isequal(v1::ExpParameter, v2::ExpParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ExpParameter, v2::ExpParameter) = ProtoBuf.protoeq(v1, v2)

type FlattenParameter
    axis::Int32
    end_axis::Int32
    FlattenParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type FlattenParameter
const __val_FlattenParameter = @compat Dict(:axis => 1, :end_axis => -1)
meta(t::Type{FlattenParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_FlattenParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::FlattenParameter) = ProtoBuf.protohash(v)
isequal(v1::FlattenParameter, v2::FlattenParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::FlattenParameter, v2::FlattenParameter) = ProtoBuf.protoeq(v1, v2)

type HDF5DataParameter
    source::AbstractString
    batch_size::UInt32
    shuffle::Bool
    HDF5DataParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type HDF5DataParameter
const __val_HDF5DataParameter = @compat Dict(:shuffle => false)
meta(t::Type{HDF5DataParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_HDF5DataParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::HDF5DataParameter) = ProtoBuf.protohash(v)
isequal(v1::HDF5DataParameter, v2::HDF5DataParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::HDF5DataParameter, v2::HDF5DataParameter) = ProtoBuf.protoeq(v1, v2)

type HDF5OutputParameter
    file_name::AbstractString
    HDF5OutputParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type HDF5OutputParameter
hash(v::HDF5OutputParameter) = ProtoBuf.protohash(v)
isequal(v1::HDF5OutputParameter, v2::HDF5OutputParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::HDF5OutputParameter, v2::HDF5OutputParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_HingeLossParameter_Norm <: ProtoEnum
    L1::Int32
    L2::Int32
    __enum_HingeLossParameter_Norm() = new(1,2)
end #type __enum_HingeLossParameter_Norm
const HingeLossParameter_Norm = __enum_HingeLossParameter_Norm()

type HingeLossParameter
    norm::Int32
    HingeLossParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type HingeLossParameter
const __val_HingeLossParameter = @compat Dict(:norm => HingeLossParameter_Norm.L1)
meta(t::Type{HingeLossParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_HingeLossParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::HingeLossParameter) = ProtoBuf.protohash(v)
isequal(v1::HingeLossParameter, v2::HingeLossParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::HingeLossParameter, v2::HingeLossParameter) = ProtoBuf.protoeq(v1, v2)

type ImageDataParameter
    source::AbstractString
    batch_size::UInt32
    rand_skip::UInt32
    shuffle::Bool
    new_height::UInt32
    new_width::UInt32
    is_color::Bool
    scale::Float32
    mean_file::AbstractString
    crop_size::UInt32
    mirror::Bool
    root_folder::AbstractString
    ImageDataParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ImageDataParameter
const __val_ImageDataParameter = @compat Dict(:batch_size => 1, :rand_skip => 0, :shuffle => false, :new_height => 0, :new_width => 0, :is_color => true, :scale => 1, :crop_size => 0, :mirror => false)
const __fnum_ImageDataParameter = Int[1,4,7,8,9,10,11,2,3,5,6,12]
meta(t::Type{ImageDataParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ImageDataParameter, __val_ImageDataParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ImageDataParameter) = ProtoBuf.protohash(v)
isequal(v1::ImageDataParameter, v2::ImageDataParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ImageDataParameter, v2::ImageDataParameter) = ProtoBuf.protoeq(v1, v2)

type InfogainLossParameter
    source::AbstractString
    InfogainLossParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type InfogainLossParameter
hash(v::InfogainLossParameter) = ProtoBuf.protohash(v)
isequal(v1::InfogainLossParameter, v2::InfogainLossParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::InfogainLossParameter, v2::InfogainLossParameter) = ProtoBuf.protoeq(v1, v2)

type InnerProductParameter
    num_output::UInt32
    bias_term::Bool
    weight_filler::FillerParameter
    bias_filler::FillerParameter
    axis::Int32
    transpose::Bool
    InnerProductParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type InnerProductParameter
const __val_InnerProductParameter = @compat Dict(:bias_term => true, :axis => 1, :transpose => false)
meta(t::Type{InnerProductParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_InnerProductParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::InnerProductParameter) = ProtoBuf.protohash(v)
isequal(v1::InnerProductParameter, v2::InnerProductParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::InnerProductParameter, v2::InnerProductParameter) = ProtoBuf.protoeq(v1, v2)

type InputParameter
    shape::Array{BlobShape,1}
    InputParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type InputParameter
hash(v::InputParameter) = ProtoBuf.protohash(v)
isequal(v1::InputParameter, v2::InputParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::InputParameter, v2::InputParameter) = ProtoBuf.protoeq(v1, v2)

type LogParameter
    base::Float32
    scale::Float32
    shift::Float32
    LogParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type LogParameter
const __val_LogParameter = @compat Dict(:base => -1, :scale => 1, :shift => 0)
meta(t::Type{LogParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_LogParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::LogParameter) = ProtoBuf.protohash(v)
isequal(v1::LogParameter, v2::LogParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::LogParameter, v2::LogParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_LRNParameter_NormRegion <: ProtoEnum
    ACROSS_CHANNELS::Int32
    WITHIN_CHANNEL::Int32
    __enum_LRNParameter_NormRegion() = new(0,1)
end #type __enum_LRNParameter_NormRegion
const LRNParameter_NormRegion = __enum_LRNParameter_NormRegion()

type __enum_LRNParameter_Engine <: ProtoEnum
    DEFAULT::Int32
    CAFFE::Int32
    CUDNN::Int32
    __enum_LRNParameter_Engine() = new(0,1,2)
end #type __enum_LRNParameter_Engine
const LRNParameter_Engine = __enum_LRNParameter_Engine()

type LRNParameter
    local_size::UInt32
    alpha::Float32
    beta::Float32
    norm_region::Int32
    k::Float32
    engine::Int32
    LRNParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type LRNParameter
const __val_LRNParameter = @compat Dict(:local_size => 5, :alpha => 1, :beta => 0.75, :norm_region => LRNParameter_NormRegion.ACROSS_CHANNELS, :k => 1, :engine => LRNParameter_Engine.DEFAULT)
meta(t::Type{LRNParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_LRNParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::LRNParameter) = ProtoBuf.protohash(v)
isequal(v1::LRNParameter, v2::LRNParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::LRNParameter, v2::LRNParameter) = ProtoBuf.protoeq(v1, v2)

type MemoryDataParameter
    batch_size::UInt32
    channels::UInt32
    height::UInt32
    width::UInt32
    MemoryDataParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type MemoryDataParameter
hash(v::MemoryDataParameter) = ProtoBuf.protohash(v)
isequal(v1::MemoryDataParameter, v2::MemoryDataParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::MemoryDataParameter, v2::MemoryDataParameter) = ProtoBuf.protoeq(v1, v2)

type MVNParameter
    normalize_variance::Bool
    across_channels::Bool
    eps::Float32
    MVNParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type MVNParameter
const __val_MVNParameter = @compat Dict(:normalize_variance => true, :across_channels => false, :eps => 1e-09)
meta(t::Type{MVNParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_MVNParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::MVNParameter) = ProtoBuf.protohash(v)
isequal(v1::MVNParameter, v2::MVNParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::MVNParameter, v2::MVNParameter) = ProtoBuf.protoeq(v1, v2)

type ParameterParameter
    shape::BlobShape
    ParameterParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ParameterParameter
hash(v::ParameterParameter) = ProtoBuf.protohash(v)
isequal(v1::ParameterParameter, v2::ParameterParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ParameterParameter, v2::ParameterParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_PoolingParameter_PoolMethod <: ProtoEnum
    MAX::Int32
    AVE::Int32
    STOCHASTIC::Int32
    __enum_PoolingParameter_PoolMethod() = new(0,1,2)
end #type __enum_PoolingParameter_PoolMethod
const PoolingParameter_PoolMethod = __enum_PoolingParameter_PoolMethod()

type __enum_PoolingParameter_Engine <: ProtoEnum
    DEFAULT::Int32
    CAFFE::Int32
    CUDNN::Int32
    __enum_PoolingParameter_Engine() = new(0,1,2)
end #type __enum_PoolingParameter_Engine
const PoolingParameter_Engine = __enum_PoolingParameter_Engine()

type PoolingParameter
    pool::Int32
    pad::UInt32
    pad_h::UInt32
    pad_w::UInt32
    kernel_size::UInt32
    kernel_h::UInt32
    kernel_w::UInt32
    stride::UInt32
    stride_h::UInt32
    stride_w::UInt32
    engine::Int32
    global_pooling::Bool
    PoolingParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type PoolingParameter
const __val_PoolingParameter = @compat Dict(:pool => PoolingParameter_PoolMethod.MAX, :pad => 0, :pad_h => 0, :pad_w => 0, :stride => 1, :engine => PoolingParameter_Engine.DEFAULT, :global_pooling => false)
const __fnum_PoolingParameter = Int[1,4,9,10,2,5,6,3,7,8,11,12]
meta(t::Type{PoolingParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_PoolingParameter, __val_PoolingParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::PoolingParameter) = ProtoBuf.protohash(v)
isequal(v1::PoolingParameter, v2::PoolingParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::PoolingParameter, v2::PoolingParameter) = ProtoBuf.protoeq(v1, v2)

type PowerParameter
    power::Float32
    scale::Float32
    shift::Float32
    PowerParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type PowerParameter
const __val_PowerParameter = @compat Dict(:power => 1, :scale => 1, :shift => 0)
meta(t::Type{PowerParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_PowerParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::PowerParameter) = ProtoBuf.protohash(v)
isequal(v1::PowerParameter, v2::PowerParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::PowerParameter, v2::PowerParameter) = ProtoBuf.protoeq(v1, v2)

type PythonParameter
    _module::AbstractString
    layer::AbstractString
    param_str::AbstractString
    share_in_parallel::Bool
    PythonParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type PythonParameter
const __val_PythonParameter = @compat Dict(:share_in_parallel => false)
meta(t::Type{PythonParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_PythonParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::PythonParameter) = ProtoBuf.protohash(v)
isequal(v1::PythonParameter, v2::PythonParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::PythonParameter, v2::PythonParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_ReductionParameter_ReductionOp <: ProtoEnum
    SUM::Int32
    ASUM::Int32
    SUMSQ::Int32
    MEAN::Int32
    __enum_ReductionParameter_ReductionOp() = new(1,2,3,4)
end #type __enum_ReductionParameter_ReductionOp
const ReductionParameter_ReductionOp = __enum_ReductionParameter_ReductionOp()

type ReductionParameter
    operation::Int32
    axis::Int32
    coeff::Float32
    ReductionParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ReductionParameter
const __val_ReductionParameter = @compat Dict(:operation => ReductionParameter_ReductionOp.SUM, :axis => 0, :coeff => 1)
meta(t::Type{ReductionParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ReductionParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ReductionParameter) = ProtoBuf.protohash(v)
isequal(v1::ReductionParameter, v2::ReductionParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ReductionParameter, v2::ReductionParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_ReLUParameter_Engine <: ProtoEnum
    DEFAULT::Int32
    CAFFE::Int32
    CUDNN::Int32
    __enum_ReLUParameter_Engine() = new(0,1,2)
end #type __enum_ReLUParameter_Engine
const ReLUParameter_Engine = __enum_ReLUParameter_Engine()

type ReLUParameter
    negative_slope::Float32
    engine::Int32
    ReLUParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ReLUParameter
const __val_ReLUParameter = @compat Dict(:negative_slope => 0, :engine => ReLUParameter_Engine.DEFAULT)
meta(t::Type{ReLUParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ReLUParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ReLUParameter) = ProtoBuf.protohash(v)
isequal(v1::ReLUParameter, v2::ReLUParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ReLUParameter, v2::ReLUParameter) = ProtoBuf.protoeq(v1, v2)

type ReshapeParameter
    shape::BlobShape
    axis::Int32
    num_axes::Int32
    ReshapeParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ReshapeParameter
const __val_ReshapeParameter = @compat Dict(:axis => 0, :num_axes => -1)
meta(t::Type{ReshapeParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ReshapeParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ReshapeParameter) = ProtoBuf.protohash(v)
isequal(v1::ReshapeParameter, v2::ReshapeParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ReshapeParameter, v2::ReshapeParameter) = ProtoBuf.protoeq(v1, v2)

type ScaleParameter
    axis::Int32
    num_axes::Int32
    filler::FillerParameter
    bias_term::Bool
    bias_filler::FillerParameter
    ScaleParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ScaleParameter
const __val_ScaleParameter = @compat Dict(:axis => 1, :num_axes => 1, :bias_term => false)
meta(t::Type{ScaleParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ScaleParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ScaleParameter) = ProtoBuf.protohash(v)
isequal(v1::ScaleParameter, v2::ScaleParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ScaleParameter, v2::ScaleParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_SigmoidParameter_Engine <: ProtoEnum
    DEFAULT::Int32
    CAFFE::Int32
    CUDNN::Int32
    __enum_SigmoidParameter_Engine() = new(0,1,2)
end #type __enum_SigmoidParameter_Engine
const SigmoidParameter_Engine = __enum_SigmoidParameter_Engine()

type SigmoidParameter
    engine::Int32
    SigmoidParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SigmoidParameter
const __val_SigmoidParameter = @compat Dict(:engine => SigmoidParameter_Engine.DEFAULT)
meta(t::Type{SigmoidParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_SigmoidParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::SigmoidParameter) = ProtoBuf.protohash(v)
isequal(v1::SigmoidParameter, v2::SigmoidParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::SigmoidParameter, v2::SigmoidParameter) = ProtoBuf.protoeq(v1, v2)

type SliceParameter
    axis::Int32
    slice_point::Array{UInt32,1}
    slice_dim::UInt32
    SliceParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SliceParameter
const __val_SliceParameter = @compat Dict(:axis => 1, :slice_dim => 1)
const __fnum_SliceParameter = Int[3,2,1]
meta(t::Type{SliceParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_SliceParameter, __val_SliceParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::SliceParameter) = ProtoBuf.protohash(v)
isequal(v1::SliceParameter, v2::SliceParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::SliceParameter, v2::SliceParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_SoftmaxParameter_Engine <: ProtoEnum
    DEFAULT::Int32
    CAFFE::Int32
    CUDNN::Int32
    __enum_SoftmaxParameter_Engine() = new(0,1,2)
end #type __enum_SoftmaxParameter_Engine
const SoftmaxParameter_Engine = __enum_SoftmaxParameter_Engine()

type SoftmaxParameter
    engine::Int32
    axis::Int32
    SoftmaxParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SoftmaxParameter
const __val_SoftmaxParameter = @compat Dict(:engine => SoftmaxParameter_Engine.DEFAULT, :axis => 1)
meta(t::Type{SoftmaxParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_SoftmaxParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::SoftmaxParameter) = ProtoBuf.protohash(v)
isequal(v1::SoftmaxParameter, v2::SoftmaxParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::SoftmaxParameter, v2::SoftmaxParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_TanHParameter_Engine <: ProtoEnum
    DEFAULT::Int32
    CAFFE::Int32
    CUDNN::Int32
    __enum_TanHParameter_Engine() = new(0,1,2)
end #type __enum_TanHParameter_Engine
const TanHParameter_Engine = __enum_TanHParameter_Engine()

type TanHParameter
    engine::Int32
    TanHParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TanHParameter
const __val_TanHParameter = @compat Dict(:engine => TanHParameter_Engine.DEFAULT)
meta(t::Type{TanHParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_TanHParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::TanHParameter) = ProtoBuf.protohash(v)
isequal(v1::TanHParameter, v2::TanHParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::TanHParameter, v2::TanHParameter) = ProtoBuf.protoeq(v1, v2)

type TileParameter
    axis::Int32
    tiles::Int32
    TileParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TileParameter
const __val_TileParameter = @compat Dict(:axis => 1)
meta(t::Type{TileParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_TileParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::TileParameter) = ProtoBuf.protohash(v)
isequal(v1::TileParameter, v2::TileParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::TileParameter, v2::TileParameter) = ProtoBuf.protoeq(v1, v2)

type ThresholdParameter
    threshold::Float32
    ThresholdParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ThresholdParameter
const __val_ThresholdParameter = @compat Dict(:threshold => 0)
meta(t::Type{ThresholdParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_ThresholdParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::ThresholdParameter) = ProtoBuf.protohash(v)
isequal(v1::ThresholdParameter, v2::ThresholdParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::ThresholdParameter, v2::ThresholdParameter) = ProtoBuf.protoeq(v1, v2)

type WindowDataParameter
    source::AbstractString
    scale::Float32
    mean_file::AbstractString
    batch_size::UInt32
    crop_size::UInt32
    mirror::Bool
    fg_threshold::Float32
    bg_threshold::Float32
    fg_fraction::Float32
    context_pad::UInt32
    crop_mode::AbstractString
    cache_images::Bool
    root_folder::AbstractString
    WindowDataParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type WindowDataParameter
const __val_WindowDataParameter = @compat Dict(:scale => 1, :crop_size => 0, :mirror => false, :fg_threshold => 0.5, :bg_threshold => 0.5, :fg_fraction => 0.25, :context_pad => 0, :crop_mode => "warp", :cache_images => false)
meta(t::Type{WindowDataParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_WindowDataParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::WindowDataParameter) = ProtoBuf.protohash(v)
isequal(v1::WindowDataParameter, v2::WindowDataParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::WindowDataParameter, v2::WindowDataParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_SPPParameter_PoolMethod <: ProtoEnum
    MAX::Int32
    AVE::Int32
    STOCHASTIC::Int32
    __enum_SPPParameter_PoolMethod() = new(0,1,2)
end #type __enum_SPPParameter_PoolMethod
const SPPParameter_PoolMethod = __enum_SPPParameter_PoolMethod()

type __enum_SPPParameter_Engine <: ProtoEnum
    DEFAULT::Int32
    CAFFE::Int32
    CUDNN::Int32
    __enum_SPPParameter_Engine() = new(0,1,2)
end #type __enum_SPPParameter_Engine
const SPPParameter_Engine = __enum_SPPParameter_Engine()

type SPPParameter
    pyramid_height::UInt32
    pool::Int32
    engine::Int32
    SPPParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SPPParameter
const __val_SPPParameter = @compat Dict(:pool => SPPParameter_PoolMethod.MAX, :engine => SPPParameter_Engine.DEFAULT)
const __fnum_SPPParameter = Int[1,2,6]
meta(t::Type{SPPParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_SPPParameter, __val_SPPParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::SPPParameter) = ProtoBuf.protohash(v)
isequal(v1::SPPParameter, v2::SPPParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::SPPParameter, v2::SPPParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_V0LayerParameter_PoolMethod <: ProtoEnum
    MAX::Int32
    AVE::Int32
    STOCHASTIC::Int32
    __enum_V0LayerParameter_PoolMethod() = new(0,1,2)
end #type __enum_V0LayerParameter_PoolMethod
const V0LayerParameter_PoolMethod = __enum_V0LayerParameter_PoolMethod()

type V0LayerParameter
    name::AbstractString
    _type::AbstractString
    num_output::UInt32
    biasterm::Bool
    weight_filler::FillerParameter
    bias_filler::FillerParameter
    pad::UInt32
    kernelsize::UInt32
    group::UInt32
    stride::UInt32
    pool::Int32
    dropout_ratio::Float32
    local_size::UInt32
    alpha::Float32
    beta::Float32
    k::Float32
    source::AbstractString
    scale::Float32
    meanfile::AbstractString
    batchsize::UInt32
    cropsize::UInt32
    mirror::Bool
    blobs::Array{BlobProto,1}
    blobs_lr::Array{Float32,1}
    weight_decay::Array{Float32,1}
    rand_skip::UInt32
    det_fg_threshold::Float32
    det_bg_threshold::Float32
    det_fg_fraction::Float32
    det_context_pad::UInt32
    det_crop_mode::AbstractString
    new_num::Int32
    new_channels::Int32
    new_height::Int32
    new_width::Int32
    shuffle_images::Bool
    concat_dim::UInt32
    hdf5_output_param::HDF5OutputParameter
    V0LayerParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type V0LayerParameter
const __val_V0LayerParameter = @compat Dict(:biasterm => true, :pad => 0, :group => 1, :stride => 1, :pool => V0LayerParameter_PoolMethod.MAX, :dropout_ratio => 0.5, :local_size => 5, :alpha => 1, :beta => 0.75, :k => 1, :scale => 1, :cropsize => 0, :mirror => false, :rand_skip => 0, :det_fg_threshold => 0.5, :det_bg_threshold => 0.5, :det_fg_fraction => 0.25, :det_context_pad => 0, :det_crop_mode => "warp", :new_num => 0, :new_channels => 0, :new_height => 0, :new_width => 0, :shuffle_images => false, :concat_dim => 1)
const __fnum_V0LayerParameter = Int[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,16,17,18,19,20,21,50,51,52,53,54,55,56,58,59,60,61,62,63,64,65,1001]
meta(t::Type{V0LayerParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_V0LayerParameter, __val_V0LayerParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::V0LayerParameter) = ProtoBuf.protohash(v)
isequal(v1::V0LayerParameter, v2::V0LayerParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::V0LayerParameter, v2::V0LayerParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_V1LayerParameter_LayerType <: ProtoEnum
    NONE::Int32
    ABSVAL::Int32
    ACCURACY::Int32
    ARGMAX::Int32
    BNLL::Int32
    CONCAT::Int32
    CONTRASTIVE_LOSS::Int32
    CONVOLUTION::Int32
    DATA::Int32
    DECONVOLUTION::Int32
    DROPOUT::Int32
    DUMMY_DATA::Int32
    EUCLIDEAN_LOSS::Int32
    ELTWISE::Int32
    EXP::Int32
    FLATTEN::Int32
    HDF5_DATA::Int32
    HDF5_OUTPUT::Int32
    HINGE_LOSS::Int32
    IM2COL::Int32
    IMAGE_DATA::Int32
    INFOGAIN_LOSS::Int32
    INNER_PRODUCT::Int32
    LRN::Int32
    MEMORY_DATA::Int32
    MULTINOMIAL_LOGISTIC_LOSS::Int32
    MVN::Int32
    POOLING::Int32
    POWER::Int32
    RELU::Int32
    SIGMOID::Int32
    SIGMOID_CROSS_ENTROPY_LOSS::Int32
    SILENCE::Int32
    SOFTMAX::Int32
    SOFTMAX_LOSS::Int32
    SPLIT::Int32
    SLICE::Int32
    TANH::Int32
    WINDOW_DATA::Int32
    THRESHOLD::Int32
    __enum_V1LayerParameter_LayerType() = new(0,35,1,30,2,3,37,4,5,39,6,32,7,25,38,8,9,10,28,11,12,13,14,15,29,16,34,17,26,18,19,27,36,20,21,22,33,23,24,31)
end #type __enum_V1LayerParameter_LayerType
const V1LayerParameter_LayerType = __enum_V1LayerParameter_LayerType()

type __enum_V1LayerParameter_DimCheckMode <: ProtoEnum
    STRICT::Int32
    PERMISSIVE::Int32
    __enum_V1LayerParameter_DimCheckMode() = new(0,1)
end #type __enum_V1LayerParameter_DimCheckMode
const V1LayerParameter_DimCheckMode = __enum_V1LayerParameter_DimCheckMode()

type V1LayerParameter
    bottom::Array{AbstractString,1}
    top::Array{AbstractString,1}
    name::AbstractString
    include::Array{NetStateRule,1}
    exclude::Array{NetStateRule,1}
    _type::Int32
    blobs::Array{BlobProto,1}
    param::Array{AbstractString,1}
    blob_share_mode::Array{Int32,1}
    blobs_lr::Array{Float32,1}
    weight_decay::Array{Float32,1}
    loss_weight::Array{Float32,1}
    accuracy_param::AccuracyParameter
    argmax_param::ArgMaxParameter
    concat_param::ConcatParameter
    contrastive_loss_param::ContrastiveLossParameter
    convolution_param::ConvolutionParameter
    data_param::DataParameter
    dropout_param::DropoutParameter
    dummy_data_param::DummyDataParameter
    eltwise_param::EltwiseParameter
    exp_param::ExpParameter
    hdf5_data_param::HDF5DataParameter
    hdf5_output_param::HDF5OutputParameter
    hinge_loss_param::HingeLossParameter
    image_data_param::ImageDataParameter
    infogain_loss_param::InfogainLossParameter
    inner_product_param::InnerProductParameter
    lrn_param::LRNParameter
    memory_data_param::MemoryDataParameter
    mvn_param::MVNParameter
    pooling_param::PoolingParameter
    power_param::PowerParameter
    relu_param::ReLUParameter
    sigmoid_param::SigmoidParameter
    softmax_param::SoftmaxParameter
    slice_param::SliceParameter
    tanh_param::TanHParameter
    threshold_param::ThresholdParameter
    window_data_param::WindowDataParameter
    transform_param::TransformationParameter
    loss_param::LossParameter
    layer::V0LayerParameter
    V1LayerParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type V1LayerParameter
const __fnum_V1LayerParameter = Int[2,3,4,32,33,5,6,1001,1002,7,8,35,27,23,9,40,10,11,12,26,24,41,13,14,29,15,16,17,18,22,34,19,21,30,38,39,31,37,25,20,36,42,1]
meta(t::Type{V1LayerParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_V1LayerParameter, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::V1LayerParameter) = ProtoBuf.protohash(v)
isequal(v1::V1LayerParameter, v2::V1LayerParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::V1LayerParameter, v2::V1LayerParameter) = ProtoBuf.protoeq(v1, v2)

type PReLUParameter
    filler::FillerParameter
    channel_shared::Bool
    PReLUParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type PReLUParameter
const __val_PReLUParameter = @compat Dict(:channel_shared => false)
meta(t::Type{PReLUParameter}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, __val_PReLUParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::PReLUParameter) = ProtoBuf.protohash(v)
isequal(v1::PReLUParameter, v2::PReLUParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::PReLUParameter, v2::PReLUParameter) = ProtoBuf.protoeq(v1, v2)

type LayerParameter
    name::AbstractString
    _type::AbstractString
    bottom::Array{AbstractString,1}
    top::Array{AbstractString,1}
    phase::Int32
    loss_weight::Array{Float32,1}
    param::Array{ParamSpec,1}
    blobs::Array{BlobProto,1}
    propagate_down::Array{Bool,1}
    include::Array{NetStateRule,1}
    exclude::Array{NetStateRule,1}
    transform_param::TransformationParameter
    loss_param::LossParameter
    accuracy_param::AccuracyParameter
    argmax_param::ArgMaxParameter
    batch_norm_param::BatchNormParameter
    bias_param::BiasParameter
    concat_param::ConcatParameter
    contrastive_loss_param::ContrastiveLossParameter
    convolution_param::ConvolutionParameter
    crop_param::CropParameter
    data_param::DataParameter
    dropout_param::DropoutParameter
    dummy_data_param::DummyDataParameter
    eltwise_param::EltwiseParameter
    elu_param::ELUParameter
    embed_param::EmbedParameter
    exp_param::ExpParameter
    flatten_param::FlattenParameter
    hdf5_data_param::HDF5DataParameter
    hdf5_output_param::HDF5OutputParameter
    hinge_loss_param::HingeLossParameter
    image_data_param::ImageDataParameter
    infogain_loss_param::InfogainLossParameter
    inner_product_param::InnerProductParameter
    input_param::InputParameter
    log_param::LogParameter
    lrn_param::LRNParameter
    memory_data_param::MemoryDataParameter
    mvn_param::MVNParameter
    parameter_param::ParameterParameter
    pooling_param::PoolingParameter
    power_param::PowerParameter
    prelu_param::PReLUParameter
    python_param::PythonParameter
    reduction_param::ReductionParameter
    relu_param::ReLUParameter
    reshape_param::ReshapeParameter
    scale_param::ScaleParameter
    sigmoid_param::SigmoidParameter
    softmax_param::SoftmaxParameter
    spp_param::SPPParameter
    slice_param::SliceParameter
    tanh_param::TanHParameter
    threshold_param::ThresholdParameter
    tile_param::TileParameter
    window_data_param::WindowDataParameter
    LayerParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type LayerParameter
const __fnum_LayerParameter = Int[1,2,3,4,10,5,6,7,11,8,9,100,101,102,103,139,141,104,105,106,144,107,108,109,110,140,137,111,135,112,113,114,115,116,117,143,134,118,119,120,145,121,122,131,130,136,123,133,142,124,125,132,126,127,128,138,129]
meta(t::Type{LayerParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_LayerParameter, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::LayerParameter) = ProtoBuf.protohash(v)
isequal(v1::LayerParameter, v2::LayerParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::LayerParameter, v2::LayerParameter) = ProtoBuf.protoeq(v1, v2)

type NetParameter
    name::AbstractString
    input::Array{AbstractString,1}
    input_shape::Array{BlobShape,1}
    input_dim::Array{Int32,1}
    force_backward::Bool
    state::NetState
    debug_info::Bool
    layer::Array{LayerParameter,1}
    layers::Array{V1LayerParameter,1}
    NetParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NetParameter
const __val_NetParameter = @compat Dict(:force_backward => false, :debug_info => false)
const __fnum_NetParameter = Int[1,3,8,4,5,6,7,100,2]
meta(t::Type{NetParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_NetParameter, __val_NetParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::NetParameter) = ProtoBuf.protohash(v)
isequal(v1::NetParameter, v2::NetParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::NetParameter, v2::NetParameter) = ProtoBuf.protoeq(v1, v2)

type __enum_SolverParameter_SnapshotFormat <: ProtoEnum
    HDF5::Int32
    BINARYPROTO::Int32
    __enum_SolverParameter_SnapshotFormat() = new(0,1)
end #type __enum_SolverParameter_SnapshotFormat
const SolverParameter_SnapshotFormat = __enum_SolverParameter_SnapshotFormat()

type __enum_SolverParameter_SolverMode <: ProtoEnum
    CPU::Int32
    GPU::Int32
    __enum_SolverParameter_SolverMode() = new(0,1)
end #type __enum_SolverParameter_SolverMode
const SolverParameter_SolverMode = __enum_SolverParameter_SolverMode()

type __enum_SolverParameter_SolverType <: ProtoEnum
    SGD::Int32
    NESTEROV::Int32
    ADAGRAD::Int32
    RMSPROP::Int32
    ADADELTA::Int32
    ADAM::Int32
    __enum_SolverParameter_SolverType() = new(0,1,2,3,4,5)
end #type __enum_SolverParameter_SolverType
const SolverParameter_SolverType = __enum_SolverParameter_SolverType()

type SolverParameter
    net::AbstractString
    net_param::NetParameter
    train_net::AbstractString
    test_net::Array{AbstractString,1}
    train_net_param::NetParameter
    test_net_param::Array{NetParameter,1}
    train_state::NetState
    test_state::Array{NetState,1}
    test_iter::Array{Int32,1}
    test_interval::Int32
    test_compute_loss::Bool
    test_initialization::Bool
    base_lr::Float32
    display::Int32
    average_loss::Int32
    max_iter::Int32
    iter_size::Int32
    lr_policy::AbstractString
    gamma::Float32
    power::Float32
    momentum::Float32
    weight_decay::Float32
    regularization_type::AbstractString
    stepsize::Int32
    stepvalue::Array{Int32,1}
    clip_gradients::Float32
    snapshot::Int32
    snapshot_prefix::AbstractString
    snapshot_diff::Bool
    snapshot_format::Int32
    solver_mode::Int32
    device_id::Int32
    random_seed::Int64
    _type::AbstractString
    delta::Float32
    momentum2::Float32
    rms_decay::Float32
    debug_info::Bool
    snapshot_after_train::Bool
    solver_type::Int32
    SolverParameter(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SolverParameter
const __val_SolverParameter = @compat Dict(:test_interval => 0, :test_compute_loss => false, :test_initialization => true, :average_loss => 1, :iter_size => 1, :regularization_type => "L2", :clip_gradients => -1, :snapshot => 0, :snapshot_diff => false, :snapshot_format => SolverParameter_SnapshotFormat.BINARYPROTO, :solver_mode => SolverParameter_SolverMode.GPU, :device_id => 0, :random_seed => -1, :_type => "SGD", :delta => 1e-08, :momentum2 => 0.999, :debug_info => false, :snapshot_after_train => true, :solver_type => SolverParameter_SolverType.SGD)
const __fnum_SolverParameter = Int[24,25,1,2,21,22,26,27,3,4,19,32,5,6,33,7,36,8,9,10,11,12,29,13,34,35,14,15,16,37,17,18,20,40,31,39,38,23,28,30]
meta(t::Type{SolverParameter}) = meta(t, ProtoBuf.DEF_REQ, __fnum_SolverParameter, __val_SolverParameter, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES)
hash(v::SolverParameter) = ProtoBuf.protohash(v)
isequal(v1::SolverParameter, v2::SolverParameter) = ProtoBuf.protoisequal(v1, v2)
==(v1::SolverParameter, v2::SolverParameter) = ProtoBuf.protoeq(v1, v2)

export Phase, BlobShape, BlobProto, BlobProtoVector, Datum, FillerParameter_VarianceNorm, FillerParameter, NetParameter, SolverParameter_SnapshotFormat, SolverParameter_SolverMode, SolverParameter_SolverType, SolverParameter, SolverState, NetState, NetStateRule, ParamSpec_DimCheckMode, ParamSpec, LayerParameter, TransformationParameter, LossParameter_NormalizationMode, LossParameter, AccuracyParameter, ArgMaxParameter, ConcatParameter, BatchNormParameter, BiasParameter, ContrastiveLossParameter, ConvolutionParameter_Engine, ConvolutionParameter, CropParameter, DataParameter_DB, DataParameter, DropoutParameter, DummyDataParameter, EltwiseParameter_EltwiseOp, EltwiseParameter, ELUParameter, EmbedParameter, ExpParameter, FlattenParameter, HDF5DataParameter, HDF5OutputParameter, HingeLossParameter_Norm, HingeLossParameter, ImageDataParameter, InfogainLossParameter, InnerProductParameter, InputParameter, LogParameter, LRNParameter_NormRegion, LRNParameter_Engine, LRNParameter, MemoryDataParameter, MVNParameter, ParameterParameter, PoolingParameter_PoolMethod, PoolingParameter_Engine, PoolingParameter, PowerParameter, PythonParameter, ReductionParameter_ReductionOp, ReductionParameter, ReLUParameter_Engine, ReLUParameter, ReshapeParameter, ScaleParameter, SigmoidParameter_Engine, SigmoidParameter, SliceParameter, SoftmaxParameter_Engine, SoftmaxParameter, TanHParameter_Engine, TanHParameter, TileParameter, ThresholdParameter, WindowDataParameter, SPPParameter_PoolMethod, SPPParameter_Engine, SPPParameter, V1LayerParameter_LayerType, V1LayerParameter_DimCheckMode, V1LayerParameter, V0LayerParameter_PoolMethod, V0LayerParameter, PReLUParameter
