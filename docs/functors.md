# Functors

## 🔨 Concat
Concatenates arrays along the specified dimension.

- `Concat(dim)`

### 👉 Example
```julia
f = Concat(1)
x1 = Variable(rand(Float32,10,5))
x2 = Variable(rand(Float32,10,5))
y = f(x1, x2)
```

## 🔨 CrossEntropy
Computes cross-entropy between a true distribution \(p\) and the target distribution \(q\).
$$f(p,q)=-\sum_{x}p(x)\log q(x)$$

- `CrossEntropy(p, x)`

### 👉 Example
```julia
```

## 🔨 Linear
Computes linear transformation a.k.a. affine transformation.
$$f(x) = Wx + b$$
where \(W\) is a weight matrix, \(b\) is a bias vector.

- `Linear(w, b)`
- `Linear{T}(::Type{T}, xlength::Int, ylength::Int)`

### 👉 Example
```julia
```

## 🔨 LogSoftmax
$$f(x)=\frac{\exp(x_{i})}{\sum_{j}^{n}\exp(x_{j})},\;i=1,\ldots,n$$

### 👉 Example
```julia
```

## 🔨 MaxPooling

- `MaxPooling(dim)`

### 👉 Example
```julia
```

## 🔨 Window2D

- `Window(w1, w2, s1, s2, p1, p2)`
    - w1, w2: window sizes
    - s1, s2: stride sizes
    - p1, p2: padding sizes

### 👉 Example
```julia
```

## 🔨 ReLU
Rectifier linear unit.

- `ReLU()`

### 👉 Example
```julia
```
