module NMFProject

using LinearAlgebra
using Random
using Statistics
using Printf
using Base.Threads

const MAX_ITER_FIXED = 10000
const TOL_FIXED = 1e-4
const TOL_FIXED_SUB = 1e-6

export generate_matrix, generate_X_WH
export make_rule_spectral_W, make_rule_spectral_H
export nmf_multiplicative, nmf_lin_algorithm, nmf_gradient_projected, projected_gradient_lin_H
export nmf_pca_wrapper

include("utils.jl")
include("step_rules.jl")
include("algorithms/multiplicative.jl")
include("algorithms/lin.jl")
include("algorithms/spectral.jl")
include("algorithms/pca.jl")

end