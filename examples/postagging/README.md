# POS-Tagging
This is an example of pos-tagging with neural networks.  
The training/testing data in the example is [UD_English](https://github.com/UniversalDependencies/UD_English).

## Usage
First, make sure you have installed `Merlin`.  
Then, clone [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl.git) to get a dataset.
```julia
julia> Pkg.clone("https://github.com/JuliaML/MLDatasets.jl.git")
```

Then, download [pre-trained word embeddings](https://cl.naist.jp/~shindo/wordembeds_nyt100.h5) and put it in this directory.

Then, run the script:
```
julia train.jl
```
