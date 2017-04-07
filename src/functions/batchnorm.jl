export batchnorm

function batchnorm(x::Var, scale::Var, bias::Var, mean, var, istrain; epsilon=0.001)
    forward0(batchnorm, x, scale, bias, mean, var)
end

function batchnorm_training(x::Array, scale::Array, bias::Array, decay; epsilon=0.001, momentum=0.99)
    runmean = zeros(scale)
    runvar = zeros(scale)

    x64 = Array{Float64}(x)
    E = Array{T}(mean(x64,(1,2,4)))
    V = var(x64, (1,2,4))
    invstd = Array{T}((V * (length(x[:,:,1,:]) - 1) / length(x[:,:,1,:]) + epsilon).^-0.5)
    V = Array{T}(V)
    xhat = similar(x)
    for i = 1:size(x,3)
        xhat[:,:,i,:] = (x[:,:,i,:] - E[i]) * invstd[i]
    end
    runmean[:,:,:,:] = decay * E + (1 - decay) * runmean
    runvar[:,:,:,:] = decay * V + (1 - decay) * runvar

    batchnorm_forwarding(xhat, scale, bias), invstd, xhat
    #batchnorm_training!(x, scale, bias, decay, epsilon, runmean, runvar)..., runmean, runvar
end

function batchnorm_training!{T}(x::Array{T}, scale::Array{T}, bias::Array{T}, decay,
    epsilon::Float64, runmean::Array{T}, runvar::Array{T})

    x64 = Array{Float64}(x)
    E = Array{T}(mean(x64, (1,2,4)))
    V = var(x64, (1,2,4))
    invstd = Array{T}((V * (length(x[:,:,1,:]) - 1) / length(x[:,:,1,:]) + epsilon).^-0.5)
    V = Array{T}(V)
    xhat = similar(x)
    for i=1:size(x,3)
        xhat[:,:,i,:] = (x[:,:,i,:] - E[i]) * invstd[i]
    end

    runmean[:,:,:,:] = decay * E + (1 - decay) * runmean
    runvar[:,:,:,:] = decay * V + (1 - decay) * runvar

    batchnorm_forwarding(xhat, scale, bias), invstd, xhat
end

function batchnorm_inference(x::Array, scale::Array, bias::Array, estmean::Array,
    estvar::Array, epsilon)

    invstd = 1 ./ √(estvar + epsilon)
    xhat = similar(x)
    for i=1:size(x,3)
        xhat[:,:,i,:] = (x[:,:,i,:] - estmean[i]) * invstd[i]
    end
    batchnorm_forwarding(xhat, scale, bias)
end

function batchnorm_forwarding(xhat::Array, scale::Array, bias::Array)
    y = similar(xhat)
    for i=1:size(y,3)
        y[:,:,i,:] = scale[i] * xhat[:,:,i,:] + bias[i]
    end
    y
end

function ∇batchnorm!{T}(gx, gy::Array{T}, scale::Array{T}, gscale, gbias, invstd::Array{T},
    xhat::Array{T})

    gb = Array{T}(sum(Array{Float64}(gy), (1,2,4)))
    gs = Array{T}(sum(Array{Float64}(gy.*xhat), (1,2,4)))
    if gx != nothing
        for i=1:size(gx,3)
            gx[:,:,i,:] += scale[i] * invstd[i] * (gy[:,:,i,:] - (gs[i] * xhat[:,:,i,:]
                + gb[i]) / length(gx[:,:,1,:]))
        end
    end
    gscale == nothing || (gscale[:,:,:,:] += gs)
    gbias == nothing || (gbias[:,:,:,:] += gb)
end
