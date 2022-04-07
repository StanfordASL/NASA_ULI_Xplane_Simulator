## Load Julia packages
using Flux
using BSON: @load

## Load in trained NN - will need to change this depending on where you run these scripts from
@load "..\\..\\models\\hold_network.bson" m

## Define forward function for NN
function forward(m, x)
    out = m(x)
    mu = out[1, :]
    sigma_sq = softplus.(out[2, :]) .+ 1e-3
    return mu, sigma_sq
end

## Pass dummy image into NN so that it compiles before run-time
img = zeros(32, 64, 3, 1)
forward(m, img)