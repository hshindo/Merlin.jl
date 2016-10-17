using Merlin

function main()
    worddict = IntDict{String}()
    catdict = IntDict{String}()

    traindata = CoNLL.read()
    testdata = CoNLL.read()

    train()
end

main()
