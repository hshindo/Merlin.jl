#import Base.Broadcast: Broadcasted
import Base.Broadcast: Broadcasted, BroadcastStyle, ArrayStyle

BroadcastStyle(::Type{<:CuArray}) = ArrayStyle{CuArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{CuArray}}, ::Type{T}) where T
    similar(CuArray, T, length.(axes(bc)))
end

function Base.copyto!(dest::CuArray, bc::Broadcasted{Nothing})
    throw("Not implemented yet.")
end
