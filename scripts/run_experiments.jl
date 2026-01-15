using Pkg; Pkg.activate(".") # Ativa o ambiente da pasta atual
using LinearAlgebra
using Statistics
using Printf
using Random

includet("../src/NMFProject.jl")
using .NMFProject

Random.seed!(123)

models = Dict{Symbol, Function}(
    # :multiplicativo => nmf_multiplicative,
    :lin => nmf_lin_algorithm,

    # :pg_spectral_non_monotone => (X, r, W, H; kwargs...) -> nmf_gradient_projected(
    #     X, r, W, H;
    #     alpha_rule_W = make_rule_spectral_W(),
    #     alpha_rule_H = make_rule_spectral_H(),
    #     monotone = false,
    #     kwargs...
    # ),

    # :pg_spectral_monotone => (X, r, W, H; kwargs...) -> nmf_gradient_projected(
    #     X, r, W, H;
    #     alpha_rule_W = make_rule_spectral_W(),
    #     alpha_rule_H = make_rule_spectral_H(),
    #     monotone = true,
    #     kwargs...
    # ),
)

# Configuração dos Testes
num_trials = 5
dims = [100, 200, 1000]
r_val = 10
type_val = :uniform

println("=== Iniciando Bateria de Testes ===")

for n in dims
    println("\n>>> Dimensão: $n | Rank: $r_val | Tipo: $type_val")
    
    # Estruturas para guardar resultados
    stats_err = Dict(k => [] for k in keys(models))
    stats_time = Dict(k => [] for k in keys(models))

    for trial in 1:num_trials
        # print("Trial $trial... ")
        
        X = generate_matrix(n, n; type=type_val)
        m, n_ = size(X)
        W_init = rand(m, r_val)
        H_init = rand(r_val, n_)

        for (name, model) in models
            _, _, errs, t, _ = model(X, r_val, W_init, H_init)
            
            push!(stats_err[name], isempty(errs) ? NaN : errs[end])
            push!(stats_time[name], t)
        end
        println("OK.")
    end

    # println("\n--- Resultados Médios (n=$n) ---")
    for name in keys(models)
        avg_err = mean(stats_err[name])
        avg_time = mean(stats_time[name])
        @printf "%-30s | Erro: %.4e | Tempo: %.4fs\n" string(name) avg_err avg_time
    end
end