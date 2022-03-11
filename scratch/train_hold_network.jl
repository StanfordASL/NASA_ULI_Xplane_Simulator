using Flux
using Flux: update!, DataLoader
using Distributions
using HDF5

function LeNet5(; imgsize = (32, 64, 3), out_dim = 4)
    out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

    return Chain(
        Conv((5, 5), imgsize[end] => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(prod(out_conv_size), 120, relu),
        Dense(120, 84, relu),
        Dense(84, out_dim)
    )
end

function forward(m, x)
    out = m(x)
    μ = out[1, :]
    σ² = softplus.(out[2, :]) .+ 1e-3
    return μ, σ²
end

data_fn = "/scratch/smkatz/NASA_ULI/hold_data.h5"
imgs = h5read(data_fn, "color_imgs")
imgs = permutedims(imgs, [3, 2, 1, 4]) # get into h x w x c x batch
dtps = h5read(data_fn, "dtps")

X = imgs[:, :, :, 1:4900]
y = -dtps[1:4900]
Xval = imgs[:, :, :, 4901:end]
yval = -dtps[4901:end]

function training_loss(m, x, y)
    μ, σ² = forward(m, x)

    lpdf = -log.(σ²) ./ 2.0f0 .- (y .- μ) .^ 2 ./ (2.0f0 .* σ²)
    return -mean(lpdf)
end

# Parameters
batch_size = 128
nepoch = 100
lr = 1e-3

device = gpu

X = Float32.(X) |> device
y = Float32.(y) |> device
Xval = Float32.(Xval) |> device
yval = Float32.(yval) |> device

data = DataLoader((X, y), batchsize = batch_size, shuffle = true, partial = false)

# Create model
m = LeNet5(out_dim = 2) |> device

θ = Flux.params(m)
opt = ADAM(lr)

lv = training_loss(m, Xval, yval)
lt = training_loss(m, Xval, yval)
println("Epoch: 0", " Loss Train: ", lt, " Loss Val: ", lv)

for e = 1:nepoch
    for (x, y) in data
        _, back = Flux.pullback(() -> training_loss(m, x, y), θ)
        update!(opt, θ, back(1.0f0))
    end
    loss_val = training_loss(m, Xval, yval)
    loss_train = training_loss(m, Xval, yval)
    e % 1 == 0 ? println("Epoch: ", e, " Loss Train: ", loss_train, " Loss Val: ", loss_val) : nothing
end