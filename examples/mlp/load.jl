using Flora.MLP
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
p.pick = [0, 3, 9]
mean(argmax(p.m(x)) .== argmax(y)) = 0.956
p.pick = [1, 2, 5]
mean(argmax(p.m(x)) .== argmax(y)) = 0.989
=#


#=
julia> using UnicodePlots

julia> spy(reshape(tX[:,11], 28, 28)')
      Sparsity Pattern
      ┌──────────────┐
    1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ > 0
      │⠀⠀⠀⠀⠀⠀⢀⠇⣾⣷⣦⣤⣴⡄│ < 0
      │⠀⠀⠀⠀⢀⠀⠘⢰⣿⣿⣿⣿⣿⠟│
      │⢀⣀⣀⠠⠔⣪⣴⣿⣿⣿⣿⣿⣿⡇│
      │⢠⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿│
      │⡛⠿⢿⣿⣿⣿⣿⣿⠿⠿⣿⣿⣿⣿│
   28 │⠈⠉⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀│
      └──────────────┘
      1             28
         nz = 363
=#
