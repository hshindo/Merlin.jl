mutable struct Config
    train::Bool
end
const CONFIGS = [Config(true) for i=1:nthreads()]

getconfig() = CONFIGS[threadid()]
setconfig(config::Config) = CONFIGS[threadid()] = config

istrain() = getconfig().train
settrain(b::Bool) = getconfig().train = b
