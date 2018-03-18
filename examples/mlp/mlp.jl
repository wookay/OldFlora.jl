using Flora.MLP
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
    train!(:fashion, m, pick, X, Y; n=5)
    show_accuracy(m, pick, X, Y)
#    save(p)
end
#=
pick = [0, 3, 9]
mean(argmax(m(x)) .== argmax(y)) = 0.9556666666666667
pick = [1, 2, 5]
mean(argmax(m(x)) .== argmax(y)) = 0.9897777777777778
=#

for pick in picks
    X, Y = prepare(train_x, train_y, pick)
    show_accuracy(m, pick, X, Y)
end
#=
pick = [0, 3, 9]
mean(argmax(m(x)) .== argmax(y)) = 0.7202777777777778
pick = [1, 2, 5]
mean(argmax(m(x)) .== argmax(y)) = 0.9897777777777778
=#
