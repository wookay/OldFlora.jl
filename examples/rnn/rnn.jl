using Flora.RNN
using MLDatasets # FashionMNIST

# load full training set
train_x, train_y = FashionMNIST.traindata() # 60_000

picks = [
    [0, 3, 9], # 0 T-Shirt, 3 Dress, 9 Ankle boot
    [1, 2, 5],
]

m = chain(3)
for pick in picks
    X, Y = prepare(train_x, train_y, pick)
    p = train!(:fashion, m, pick, X, Y; n=3)
    show_accuracy(m, pick, X, Y)
    save(p)
end

#= 
train! n=3
pick = [0, 3, 9]
mean(argmax(m(x)) .== argmax(y)) = 0.9375
pick = [1, 2, 5]
mean(argmax(m(x)) .== argmax(y)) = 0.9864444444444445
=#
