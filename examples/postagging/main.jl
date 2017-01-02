using Merlin
using MLDatasets
using HDF5

const nepochs = 10

function train(model, train_x, train_y, test_x, test_y)
    opt = SGD()
    for epoch = 1:nepochs
        println("epoch: $(epoch)")
        opt.rate = 0.0075 / epoch
        loss = fit(train_x, train_y, model, crossentropy, opt)
        println("loss: $(loss)")

        test_z = map(x -> predict(model,x), test_x)
        acc = accuracy(test_y, test_z)
        println("test acc.: $(acc)")
        println("")
    end
end

predict(model, data) = argmax(model(data).data, 1)

function accuracy(golds::Vector{Vector{Int}}, preds::Vector{Vector{Int}})
    @assert length(golds) == length(preds)
    correct = 0
    total = 0
    for i = 1:length(golds)
        @assert length(golds[i]) == length(preds[i])
        for j = 1:length(golds[i])
            golds[i][j] == preds[i][j] && (correct += 1)
            total += 1
        end
    end
    correct / total
end

include("data.jl")
include("token.jl")
include("model.jl")

h5file = "wordembeds_nyt100.h5"
words = h5read(h5file, "s")
wordembeds = h5read(h5file,"v")
train_x, test_x = setup_data()
model = Model()
train()

#model = Model(wordembeds, charembeds, length(tagdict))
# model = Merlin.load("postagger.h5", "model")
train(5, model, train_x, train_y, test_x, test_y)
#Merlin.save("postagger.h5", "w", "model", model)

main()
