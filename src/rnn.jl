module RNN

export PickModel
export chain, prepare, train!, show_accuracy
export save, load

using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using BSON: @save, @load
using Base.Iterators: repeated
using MLDatasets # FashionMNIST
using ColorTypes: N0f8, Gray

const Img = Matrix{Gray{N0f8}}

struct PickModel
    title::Symbol
    pick::Vector{Int}
    m::Chain
end

function prepare(data_x, data_y, pick::Vector{Int})
    # 18_000
    inds = find(x->x in pick, data_y)

    imgs = Img.([data_x[:,:,i] for i in inds])
    labels = data_y[inds]

    X = hcat(float.(reshape.(imgs, :))...) |> gpu
    Y = onehotbatch(labels, pick) |> gpu
    X, Y
end

function chain(output::Int)
    Chain(
        LSTM(28^2, 32),
        Dense(32, output),
        softmax) |> gpu
end

function train!(title, m::Chain, pick::Vector{Int}, X, Y; n=5)
    function loss(x, y)
        l = crossentropy(m(x), y)
        Flux.truncate!(m) # Truncating Gradients
        l
    end
    dataset = repeated((X, Y), 20) # 200
    evalcb = () -> @show(loss(X, Y))
    opt = ADAM(params(m))

    @epochs n Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 3))
    PickModel(title, pick, m)
end

show_accuracy(m::Chain, pick::Vector{Int}, x, y) = @show pick mean(argmax(m(x)) .== argmax(y))

model_name(title::Symbol, pick::Vector{Int}, target) =
    string(title, "_rnn_", join(pick), "_", target, ".bson")

function save(p::PickModel)
    m = p.m
    Flux.reset!(m)
    name = model_name(p.title, p.pick, "model")
    @save name m
    weights = Tracker.data.(params(m)) ;
    name = model_name(p.title, p.pick, "weights")
    @save name weights
end

function load(title::Symbol, pick::Vector{Int})
    name = model_name(title, pick, "model")
    @load name m
    name = model_name(title, pick, "weights")
    @load name weights
    Flux.loadparams!(m, weights)
    PickModel(title, pick, m)
end

end # module RNN
