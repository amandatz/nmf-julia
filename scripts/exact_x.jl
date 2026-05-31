using Pkg; Pkg.activate(".")
using LinearAlgebra
using Statistics
using Printf
using Random
using Dates
using Distributions
using HypothesisTests

try
    using Revise
catch
end

includet("../src/NMFProject.jl")
using .NMFProject

# =========================================================================
# Funções auxiliares
# =========================================================================

function log_msg(io::IO, msg::String)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(io, "[$t] $msg")
    println("[$t] $msg")
end

function relative_error(X, W, H)
    return norm(X - W * H) / max(1.0, norm(X))
end

# =========================================================================
# Geração de X 
# =========================================================================

function generate_synthetic(m, n, r; seed=nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    delta = 1e-9;
    W_true = delta .+ (1 - delta) .* rand(rng, m, r)
    H_true = delta .+ (1 - delta) .* rand(rng, r, n)
    return W_true * H_true
end

# =========================================================================
# Experimento principal
# =========================================================================

function main()
    Random.seed!(42)

    models = Dict{Symbol, Function}(
        :lin            => nmf_lin_algorithm,
        :multiplicativo => nmf_multiplicative
    )

    num_trials = 30
    dims       = [(10, 10), (10, 200), (50, 100), (100, 200), (200, 500)]
    ranks      = [3, 5, 10]

    LOG_DIR = joinpath("resultados", "synthetic_recovery")
    mkpath(LOG_DIR)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    log_file_path = joinpath(LOG_DIR, "recovery_$timestamp.log")

    open(log_file_path, "w") do io
        log_msg(io, "=== EXPERIMENTO: RECUPERAÇÃO DE X SINTÉTICO ===")
        log_msg(io, "trials     = $num_trials")
        log_msg(io, "dims (m×n) = $dims")
        log_msg(io, "ranks      = $ranks")
        println(io, "="^80)

        for (m, n) in dims
            for r in ranks
                log_msg(io, "")
                log_msg(io, ">>> dim=$(m)×$(n) | rank=$r")

                stats_err      = Dict(k => Float64[] for k in keys(models))
                stats_time     = Dict(k => Float64[] for k in keys(models))
                stats_ext_iter = Dict(k => Int[]     for k in keys(models))
                stats_int_iter = Dict(k => Int[]     for k in keys(models))
                stats_restarts = Dict(k => Int[]     for k in keys(models))

                for trial in 1:num_trials
                    X = generate_synthetic(m, n, r; seed=trial*1000)

                    delta = 1e-9;
                    W_init = delta .+ (1-delta) .* rand(m, r)
                    H_init = delta .+ (1-delta) .* rand(r, n)

                    for (name, model_func) in models
                        if name == :lin
                            W, H, errs, t, ext_iters, int_iters, _, _, restarts = model_func(
                                X, r, copy(W_init), copy(H_init);
                                sub_tol=1e-5, sub_max_iter=200,
                                max_iter=2000, tol=1e-6, log_io=IOBuffer()
                            )
                            push!(stats_ext_iter[name], ext_iters)
                            push!(stats_int_iter[name], int_iters)
                            push!(stats_restarts[name], restarts)
                        else
                            W, H, errs, t, iters = model_func(
                                X, r, copy(W_init), copy(H_init);
                                max_iter=2000, tol=1e-6, log_io=IOBuffer()
                            )
                            push!(stats_ext_iter[name], iters)
                            push!(stats_int_iter[name], 0) 
                            push!(stats_restarts[name], 0) 
                        end

                        push!(stats_err[name],  relative_error(X, W, H))
                        push!(stats_time[name], t)
                    end
                    print(".")
                end
                println()

                println(io, "ALGORITMO      | ERRO MÉDIO ± IC95%   | TEMPO (s) ± IC95% | ITER EXT | ITER INT | REINÍCIOS")
                println(io, "-"^110)

                for name in sort(collect(keys(models)))
                    ed  = stats_err[name]
                    td  = stats_time[name]
                    eid = stats_ext_iter[name]
                    iid = stats_int_iter[name]
                    rd  = stats_restarts[name]

                    t_crit = quantile(TDist(length(ed)-1), 0.975)
                    ci(v)  = t_crit * std(v) / sqrt(length(v))

                    line = @sprintf(
                        "%-14s | %.3e ± %.2e | %.3fs ± %.3f | %8.1f | %8.1f | %9.1f",
                        string(name),
                        mean(ed), ci(ed),
                        mean(td), ci(td),
                        mean(eid),
                        mean(iid),
                        mean(rd)
                    )
                    log_msg(io, line)
                end

                if length(models) >= 2
                    println(io, "\nTeste t pareado (erro de reconstrução):")
                    names = collect(keys(models))
                    for i in 1:length(names), j in i+1:length(names)
                        a, b = names[i], names[j]
                        result = OneSampleTTest(stats_err[a] .- stats_err[b], 0.0)
                        @printf(io, "  %s vs %s: p-valor = %.4f\n", a, b, pvalue(result))
                    end
                end

                println(io, "-"^110)
            end
        end

        log_msg(io, "=== FINALIZADO ===")
    end
end

main()