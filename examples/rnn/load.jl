using Flora.RNN
using MLDatasets # FashionMNIST

# load full test set
test_x,  test_y  = FashionMNIST.testdata() # 10_000

picks = [
    [0, 3, 9], # 0 T-Shirt, 3 Dress, 9 Ankle boot
    [1, 2, 5],
]

for pick in picks
    p = load(:fashion, pick)
    tX, tY = prepare(test_x, test_y, pick)
    show_accuracy(p.m, p.pick, tX, tY)
end

#=
pick = [0, 3, 9]
mean(argmax(m(x)) .== argmax(y)) = 0.929
pick = [1, 2, 5]
mean(argmax(m(x)) .== argmax(y)) = 0.9866666666666667
=#
