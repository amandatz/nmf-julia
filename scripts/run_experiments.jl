using Pkg; Pkg.activate(".")
Pkg.add("Revise")
using LinearAlgebra
using Statistics
using Printf
using Random
using Dates

try
    using Revise
catch
end

includet("../src/NMFProject.jl")
using .NMFProject

function log_msg(io::IO, msg::String)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(io, "[$t] $msg")
    println("[$t] $msg")
end

function relative_error(X, W, H)
    return norm(X - W * H) / max(1.0, norm(X))
end

function main()
    Random.seed!(123)

    models = Dict{Symbol, Function}(
        :multiplicativo => nmf_multiplicative,
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

    num_trials = 20

    dims = [100,200,1000]
    # dims = [100]
    ranks = [5,10,20]

    types = [:uniform, :decaying, :equal, :ill_conditioned]

    LOG_DIR = joinpath("resultados","run_experiments")
    mkpath(LOG_DIR)

    timestamp = Dates.format(now(),"yyyy-mm-dd_HH-MM-SS")
    log_file_path = joinpath(LOG_DIR,"benchmark_$timestamp.log")

    open(log_file_path,"w") do io

        log_msg(io,"=== INICIANDO EXPERIMENTOS NMF ===")
        log_msg(io,"trials = $num_trials")
        log_msg(io,"dims   = $(dims)")
        log_msg(io,"ranks  = $(ranks)")
        log_msg(io,"types  = $(types)")

        println(io,"-"^80)

        for type_val in types
        for r_val in ranks
        for dim in dims

            log_msg(io,"")
            log_msg(io,">>> matriz=$type_val | dimensão=$dim x $dim | rank=$r_val")

            stats_err  = Dict(k=>Float64[] for k in keys(models))
            stats_time = Dict(k=>Float64[] for k in keys(models))
            stats_iter = Dict(k=>Int[] for k in keys(models))

            for trial in 1:num_trials

                X = generate_matrix(dim,dim;type=type_val)

                m,n = size(X)

                W_init = rand(m,r_val)
                H_init = rand(r_val,n)

                for (name,model_func) in models

                    W,H,errs,t,_ = model_func(
                        X,r_val,
                        copy(W_init),
                        copy(H_init);
                        max_iter=1000,
                        tol=1e-3,
                        log_io=devnull
                    )

                    err = relative_error(X,W,H)
                    it  = length(errs)

                    push!(stats_err[name],err)
                    push!(stats_time[name],t)
                    push!(stats_iter[name],it)

                end

                print(".")
            end

            println()

            println(io,"ALGORITMO | ERRO MÉDIO ± STD | TEMPO MÉDIO ± STD | ITER MÉDIA")
            println(io,"-"^80)

            for name in sort(collect(keys(models)))

                err_mean  = mean(stats_err[name])
                err_std   = std(stats_err[name])

                time_mean = mean(stats_time[name])
                time_std  = std(stats_time[name])

                iter_mean = mean(stats_iter[name])

                line = @sprintf(
                    "%-15s | %.3e ± %.3e | %.3fs ± %.3f | %.1f",
                    string(name),
                    err_mean,err_std,
                    time_mean,time_std,
                    iter_mean
                )

                log_msg(io,line)

            end

            println(io,"-"^80)

        end
        end
        end

        log_msg(io,"=== FINALIZADO ===")

    end

end

main()