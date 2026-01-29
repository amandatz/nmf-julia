using Pkg; Pkg.activate(".") 
using LinearAlgebra
using Statistics
using Printf
using Random
using Dates

includet("../src/NMFProject.jl")
using .NMFProject

function log_msg(io::IO, msg::String)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(io, "[$t] $msg")
    println("[$t] $msg") 
end

function main()
    Random.seed!(123)

    models = Dict{Symbol, Function}(
        :multiplicativo => nmf_multiplicative,
        :lin => nmf_lin_algorithm,

        :pg_spectral_non_monotone => (X, r, W, H; kwargs...) -> nmf_gradient_projected(
            X, r, W, H;
            alpha_rule_W = make_rule_spectral_W(),
            alpha_rule_H = make_rule_spectral_H(),
            monotone = false,
            kwargs...
        ),

        :pg_spectral_monotone => (X, r, W, H; kwargs...) -> nmf_gradient_projected(
            X, r, W, H;
            alpha_rule_W = make_rule_spectral_W(),
            alpha_rule_H = make_rule_spectral_H(),
            monotone = true,
            kwargs...
        ),
    )

    num_trials = 5
    dims = [100, 200, 1000]
    r_val = 10
    type_val = :uniform

    LOG_DIR = joinpath("resultados", "run_experiments")
    mkpath(LOG_DIR)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    log_file_path = joinpath(LOG_DIR, "benchmark_$timestamp.log")

    open(log_file_path, "w") do io
        log_msg(io, "=== INICIANDO BATERIA DE TESTES ===")
        log_msg(io, "SETUP: trials=$num_trials | rank=$r_val | tipo_matriz=$type_val")
        println(io, "-"^60)

        for n in dims
            log_msg(io, "\n>>> DIMENSÃO: $n x $n")
            
            stats_err = Dict(k => Float64[] for k in keys(models))
            stats_time = Dict(k => Float64[] for k in keys(models))

            for trial in 1:num_trials
                X = generate_matrix(n, n; type=type_val)
                m, n_cols = size(X)
                W_init = rand(m, r_val)
                H_init = rand(r_val, n_cols)

                for (name, model_func) in models
                    _, _, errs, t, _ = model_func(X, r_val, copy(W_init), copy(H_init); 
                                                  max_iter=500, tol=1e-4, log_io=devnull)
                    
                    push!(stats_err[name], isempty(errs) ? NaN : errs[end])
                    push!(stats_time[name], t)
                end
                print(".")
            end
            println()

            log_msg(io, "Resultados Médios para n=$n:")
            println(io, "ALGORITMO                      | ERRO MÉDIO   | TEMPO MÉDIO")
            println(io, "-"^60)
            
            for name in sort(collect(keys(models)))
                avg_err = mean(stats_err[name])
                avg_time = mean(stats_time[name])
                
                line = @sprintf("%-30s | %.4e | %.4fs", string(name), avg_err, avg_time)
                log_msg(io, line)
            end
            println(io, "-"^60)
        end

        log_msg(io, "\n=== FINALIZADO ===")
    end
end

main()