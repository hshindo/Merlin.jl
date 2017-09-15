# Scripts

## Convert Word Embedding Text File into HDF5
### Sample Input
```
the 0.418 0.24968 -0.41242 0.1217 0.34527
, 0.013441 0.23682 -0.16899 0.40951 0.63812
and 0.15164 0.30177 -0.16763 0.17684 0.31719
of 0.33973 -0.43478 -0.31086 -0.44999 -0.29486
```

### Run
```julia
julia embeds2h5.jl data/sample_wordembeds.txt
```
