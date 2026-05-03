using Pkg; Pkg.activate(".")
using LinearAlgebra
using Statistics
using Printf
using Random
using Dates
using HypothesisTests
using Distributions

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
# Experimento principal
# =========================================================================

function main()
    Random.seed!(123)

    models = Dict{Symbol, Function}(
        :lin => nmf_lin_algorithm,
        :multiplicativo => nmf_multiplicative
    )

    num_trials = 30
    dims = [100, 200, 1000]
    ranks = [5, 10, 20]
    types = [:uniform, :decaying, :equal, :ill_conditioned]

    LOG_DIR = joinpath("resultados", "run_experiments")
    mkpath(LOG_DIR)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    log_file_path = joinpath(LOG_DIR, "benchmark_$timestamp.log")

    open(log_file_path, "w") do io
        log_msg(io, "=== INICIANDO EXPERIMENTOS NMF ===")
        log_msg(io, "trials = $num_trials")
        log_msg(io, "dims   = $(dims)")
        log_msg(io, "ranks  = $(ranks)")
        log_msg(io, "types  = $(types)")
        println(io, "-"^80)

        for type_val in types
            for r_val in ranks
                for dim in dims
                    log_msg(io, "")
                    log_msg(io, ">>> matriz=$type_val | dimensão=$dim x $dim | rank=$r_val")

                    # Armazenar resultados por algoritmo
                    stats_err   = Dict(k => Float64[] for k in keys(models))
                    stats_time  = Dict(k => Float64[] for k in keys(models))
                    stats_iter  = Dict(k => Int[]   for k in keys(models))
                    stats_restarts = Dict(k => Int[] for k in keys(models))   # NOVO
                    stop_reason_counts = Dict(k => Dict("converged" => 0, "max_iter" => 0) for k in keys(models))
                    
                    # Para contagem de hits (apenas relevante para :lin)
                    hit_W_trials = Dict(k => Int[] for k in keys(models))
                    hit_H_trials = Dict(k => Int[] for k in keys(models))

                    for trial in 1:num_trials
                        X = generate_matrix(dim, dim; type=type_val)
                        m, n = size(X)
                        W_init = rand(m, r_val)
                        H_init = rand(r_val, n)

                        for (name, model_func) in models
                            if name == :lin
                                # Retorna 8 valores (incluindo restarts)
                                W, H, errs, t, iters, hit_W, hit_H, restarts = model_func(
                                    X, r_val,
                                    copy(W_init), copy(H_init);
                                    max_iter=1000, tol=1e-3,
                                    log_io=IOBuffer()
                                )
                                push!(hit_W_trials[name], hit_W)
                                push!(hit_H_trials[name], hit_H)
                                push!(stats_restarts[name], restarts)
                            else
                                # Retorna 5 valores (multiplicativo)
                                W, H, errs, t, iters = model_func(
                                    X, r_val,
                                    copy(W_init), copy(H_init);
                                    max_iter=1000, tol=1e-3,
                                    log_io=IOBuffer()
                                )
                                # Para multiplicativo, não há reinícios → 0
                                push!(stats_restarts[name], 0)
                            end

                            err = relative_error(X, W, H)
                            push!(stats_err[name], err)
                            push!(stats_time[name], t)
                            push!(stats_iter[name], iters)

                            if length(errs) >= 1000
                                stop_reason_counts[name]["max_iter"] += 1
                            else
                                stop_reason_counts[name]["converged"] += 1
                            end
                        end
                        print(".")
                    end
                    println()  # nova linha após pontos

                    # Calcular percentuais de hits para :lin
                    hit_W_percent = Dict(k => 0.0 for k in keys(models))
                    hit_H_percent = Dict(k => 0.0 for k in keys(models))
                    for name in keys(models)
                        if name == :lin
                            hit_W_percent[name] = 100 * count(x -> x > 0, hit_W_trials[name]) / num_trials
                            hit_H_percent[name] = 100 * count(x -> x > 0, hit_H_trials[name]) / num_trials
                        end
                    end

                    # Calcular média e desvio padrão dos reinícios
                    restart_mean = Dict(k => mean(stats_restarts[k]) for k in keys(models))
                    restart_std  = Dict(k => (length(stats_restarts[k]) > 1 ? std(stats_restarts[k]) : 0.0) for k in keys(models))

                    # Cabeçalho da tabela (nova coluna: REINÍCIOS)
                    println(io, "ALGORITMO | ERRO MÉDIO ± IC95% | TEMPO MÉDIO ± IC95% | ITER MÉDIA | CONV. % | Wmax hit % | Hmax hit % | REINÍCIOS")
                    println(io, "-"^80)

                    for name in sort(collect(keys(models)))
                        err_data = stats_err[name]
                        time_data = stats_time[name]
                        iter_data = stats_iter[name]

                        err_mean = mean(err_data)
                        err_std  = std(err_data)
                        t_val = quantile(TDist(length(err_data)-1), 0.975)
                        err_ci = t_val * err_std / sqrt(length(err_data))
                        time_mean = mean(time_data)
                        time_std  = std(time_data)
                        time_ci = t_val * time_std / sqrt(length(time_data))
                        iter_mean = mean(iter_data)

                        conv_percent = 100 * stop_reason_counts[name]["converged"] / num_trials

                        # Formatação da coluna reinícios: média ± desvio (ou só média se desvio = 0)
                        restart_str = if restart_std[name] == 0.0
                            @sprintf("%.1f", restart_mean[name])
                        else
                            @sprintf("%.1f ± %.1f", restart_mean[name], restart_std[name])
                        end

                        line = @sprintf(
                            "%-15s | %.3e ± %.3e | %.3fs ± %.3f | %.1f | %5.1f%% | %9.1f%% | %9.1f%% | %s",
                            string(name),
                            err_mean, err_ci,
                            time_mean, time_ci,
                            iter_mean,
                            conv_percent,
                            hit_W_percent[name],
                            hit_H_percent[name],
                            restart_str
                        )
                        log_msg(io, line)
                    end

                    # Teste t pareado (se houver mais de um modelo)
                    if length(models) >= 2
                        println(io, "\nTeste t pareado (diferenças de erro):")
                        names = collect(keys(models))
                        for i in 1:length(names)
                            for j in i+1:length(names)
                                a = names[i]; b = names[j]
                                diffs = stats_err[a] .- stats_err[b]
                                result = OneSampleTTest(diffs, 0.0)
                                p_val = pvalue(result)
                                println(io, "  $a vs $b: p-valor = $(round(p_val, digits=4))")
                            end
                        end
                    end

                    println(io, "-"^80)
                end
            end
        end
        log_msg(io, "=== FINALIZADO ===")
    end
end

main()