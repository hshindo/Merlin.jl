export Standardizer
import Statistics: mean, std

struct Standardizer
    meanx
    stdx
    y
end

function Standardizer(x::Matrix)
    meanx = mean(x, dims=2)
    stdx = std(x, mean=meanx, corrected=false, dims=2)
    y = (x .- meanx) ./ stdx
    Standardizer(meanx, stdx, y)
end

function (s::Standardizer)(y::Matrix)
    @assert size(y,1) == size(s.y, 1)
    x = y .* s.stdx .+ s.meanx
    x
end
