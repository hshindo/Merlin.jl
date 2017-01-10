# Julia wrapper for header: /usr/local/cuda/include/curand.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function curandCreateGenerator(generator,rng_type)
    check_curandstatus(ccall((:curandCreateGenerator,libcurand),curandStatus_t,(Ptr{curandGenerator_t},curandRngType_t),generator,rng_type))
end

function curandCreateGeneratorHost(generator,rng_type)
    check_curandstatus(ccall((:curandCreateGeneratorHost,libcurand),curandStatus_t,(Ptr{curandGenerator_t},curandRngType_t),generator,rng_type))
end

function curandDestroyGenerator(generator)
    check_curandstatus(ccall((:curandDestroyGenerator,libcurand),curandStatus_t,(curandGenerator_t,),generator))
end

function curandGetVersion(version)
    check_curandstatus(ccall((:curandGetVersion,libcurand),curandStatus_t,(Ptr{Cint},),version))
end

function curandSetStream(generator,stream)
    check_curandstatus(ccall((:curandSetStream,libcurand),curandStatus_t,(curandGenerator_t,cudaStream_t),generator,stream))
end

function curandSetPseudoRandomGeneratorSeed(generator,seed)
    check_curandstatus(ccall((:curandSetPseudoRandomGeneratorSeed,libcurand),curandStatus_t,(curandGenerator_t,Culonglong),generator,seed))
end

function curandSetGeneratorOffset(generator,offset)
    check_curandstatus(ccall((:curandSetGeneratorOffset,libcurand),curandStatus_t,(curandGenerator_t,Culonglong),generator,offset))
end

function curandSetGeneratorOrdering(generator,order)
    check_curandstatus(ccall((:curandSetGeneratorOrdering,libcurand),curandStatus_t,(curandGenerator_t,curandOrdering_t),generator,order))
end

function curandSetQuasiRandomGeneratorDimensions(generator,num_dimensions)
    check_curandstatus(ccall((:curandSetQuasiRandomGeneratorDimensions,libcurand),curandStatus_t,(curandGenerator_t,UInt32),generator,num_dimensions))
end

function curandGenerate(generator,outputPtr,num)
    check_curandstatus(ccall((:curandGenerate,libcurand),curandStatus_t,(curandGenerator_t,Ptr{UInt32},Csize_t),generator,outputPtr,num))
end

function curandGenerateLongLong(generator,outputPtr,num)
    check_curandstatus(ccall((:curandGenerateLongLong,libcurand),curandStatus_t,(curandGenerator_t,Ptr{Culonglong},Csize_t),generator,outputPtr,num))
end

function curandGenerateUniform(generator,outputPtr,num)
    check_curandstatus(ccall((:curandGenerateUniform,libcurand),curandStatus_t,(curandGenerator_t,Ptr{Cfloat},Csize_t),generator,outputPtr,num))
end

function curandGenerateUniformDouble(generator,outputPtr,num)
    check_curandstatus(ccall((:curandGenerateUniformDouble,libcurand),curandStatus_t,(curandGenerator_t,Ptr{Cdouble},Csize_t),generator,outputPtr,num))
end

function curandGenerateNormal(generator,outputPtr,n,mean,stddev)
    check_curandstatus(ccall((:curandGenerateNormal,libcurand),curandStatus_t,(curandGenerator_t,Ptr{Cfloat},Csize_t,Cfloat,Cfloat),generator,outputPtr,n,mean,stddev))
end

function curandGenerateNormalDouble(generator,outputPtr,n,mean,stddev)
    check_curandstatus(ccall((:curandGenerateNormalDouble,libcurand),curandStatus_t,(curandGenerator_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble),generator,outputPtr,n,mean,stddev))
end

function curandGenerateLogNormal(generator,outputPtr,n,mean,stddev)
    check_curandstatus(ccall((:curandGenerateLogNormal,libcurand),curandStatus_t,(curandGenerator_t,Ptr{Cfloat},Csize_t,Cfloat,Cfloat),generator,outputPtr,n,mean,stddev))
end

function curandGenerateLogNormalDouble(generator,outputPtr,n,mean,stddev)
    check_curandstatus(ccall((:curandGenerateLogNormalDouble,libcurand),curandStatus_t,(curandGenerator_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble),generator,outputPtr,n,mean,stddev))
end

function curandCreatePoissonDistribution(lambda,discrete_distribution)
    check_curandstatus(ccall((:curandCreatePoissonDistribution,libcurand),curandStatus_t,(Cdouble,Ptr{curandDiscreteDistribution_t}),lambda,discrete_distribution))
end

function curandDestroyDistribution(discrete_distribution)
    check_curandstatus(ccall((:curandDestroyDistribution,libcurand),curandStatus_t,(curandDiscreteDistribution_t,),discrete_distribution))
end

function curandGeneratePoisson(generator,outputPtr,n,lambda)
    check_curandstatus(ccall((:curandGeneratePoisson,libcurand),curandStatus_t,(curandGenerator_t,Ptr{UInt32},Csize_t,Cdouble),generator,outputPtr,n,lambda))
end

function curandGeneratePoissonMethod(generator,outputPtr,n,lambda,method)
    check_curandstatus(ccall((:curandGeneratePoissonMethod,libcurand),curandStatus_t,(curandGenerator_t,Ptr{UInt32},Csize_t,Cdouble,curandMethod_t),generator,outputPtr,n,lambda,method))
end

function curandGenerateSeeds(generator)
    check_curandstatus(ccall((:curandGenerateSeeds,libcurand),curandStatus_t,(curandGenerator_t,),generator))
end

function curandGetDirectionVectors32(vectors,set)
    check_curandstatus(ccall((:curandGetDirectionVectors32,libcurand),curandStatus_t,(Ptr{Ptr{curandDirectionVectors32_t}},curandDirectionVectorSet_t),vectors,set))
end

function curandGetScrambleConstants32(constants)
    check_curandstatus(ccall((:curandGetScrambleConstants32,libcurand),curandStatus_t,(Ptr{Ptr{UInt32}},),constants))
end

function curandGetDirectionVectors64(vectors,set)
    check_curandstatus(ccall((:curandGetDirectionVectors64,libcurand),curandStatus_t,(Ptr{Ptr{curandDirectionVectors64_t}},curandDirectionVectorSet_t),vectors,set))
end

function curandGetScrambleConstants64(constants)
    check_curandstatus(ccall((:curandGetScrambleConstants64,libcurand),curandStatus_t,(Ptr{Ptr{Culonglong}},),constants))
end
