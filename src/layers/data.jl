export Data, Param

Var(:Data)

Data(data) = Data(data, nothing, Var[])
Data() = Data(nothing)
Param(data) = Data(data, zeros(data), Var[])
