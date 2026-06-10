module NMFProject

using LinearAlgebra
using Random
using Statistics
using Printf
using Base.Threads
using Dates

export generate_matrix, generate_X_WH
export nmf_multiplicative, nmf_lin_algorithm, nmf_gradient_projected, projected_gradient_lin_H, multiplicative_H_projection

include("utils.jl")
include("step_rules.jl")
include("algorithms/multiplicative.jl")
include("algorithms/lin.jl")

end