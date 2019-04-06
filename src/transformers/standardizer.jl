export Standardizer
import Statistics: mean, std

struct Standardizer
    meanx
    stdx
    y
end

function Standardizer(x::Matrix)
    meanx = mean(x, dims=dims)
    stdx = std(x, mean=meanx, corrected=false, dims=dims)
    y = (x .- meanx) ./ stdx
    Standardizer(meanx, stdx, y)
end

function inverse(s::Standardizer, y::Matrix; dims)
    size(y,2) == size(s.y, 2)
    
end
