# NER
This is an example of NER with neural networks.  

## Usage
First, make sure you have installed `Merlin`.  
Then, clone [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl.git) to get a dataset.
```julia
julia> Pkg.clone("https://github.com/JuliaML/MLDatasets.jl.git")
```

Then, download [pre-trained word embeddings](https://cl.naist.jp/~shindo/wordembeds_nyt100.h5) and put it in this directory.

Then, run the script:
```
julia main.jl
```
