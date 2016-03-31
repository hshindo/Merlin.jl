# POS-Tagging

In this tutorial, we explain a pos-tagger based on neural network.

[Santos] dosSantos and Zadrozny,
  *Learning Character-level Representations for Part-of-Speech Tagging*,
  In Proceedings of ICML, 2014.

## Preparing the Data

CoNLL-format Penn Treebank

## Defining the Network Architecture

The neural pos-tagger consists of three types of layers:

First, we define a Token type:

```julia
  type Token
    dicts::Tuple{Dict, Dict, Dict}
    wordid::Int
    charids::Vector{Int}
    catid::Int
  end
```
```julia
  using Merlin

  type LayerSet
    wordembed::Lookup
    charseq::Sequence
    wordseq::Sequence
  end
```

Then we need to define a forward function:

```julia
  function forward(ls::LayerSet, data::Vector{Token})

  end
```julia
