export Standardizer
export inverse
using Statistics

struct Standardizer
    mean
    std
end

function Standardizer(x::Matrix)
    meanx = mean(x, dims=2)
    stdx = std(x, mean=meanx, corrected=false, dims=2)
    Standardizer(meanx, stdx)
end
function Standardizer(x::Vector)
    meanx = mean(x)
    stdx = std(x, corrected=false)
    Standardizer(meanx, stdx)
end

function (s::Standardizer)(x::Array)
    (x .- s.mean) ./ s.std
end

function inverse(s::Standardizer, y::Array)
    if ndims(y) == 2
        @assert size(y,1) == size(s.mean,1)
    end
    y .* s.std .+ s.mean
end
