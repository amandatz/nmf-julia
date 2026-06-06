using Pkg
Pkg.activate(".")
using Images, FileIO
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

# =========================================================================
# CONFIGURAÇÕES E EXECUÇÃO
# =========================================================================

const DATA_PATH = joinpath(@__DIR__, "..", "data", "att_face_dataset")
const RANKS     = [5, 10, 25, 40]
const MAX_ITER  = 500
const TOL       = 1e-4
const NUM_TRAIN_PER_PERSON = 7 
const IMG_SIZE = (112, 92)

function log_msg(io::IO, msg::String)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(io, "[$t] $msg")
    println("[$t] $msg") 
end

function rand_pos(dims...; delta=1e-9)
    return delta .+ (1 - delta) .* rand(dims...)
end

function project_new_data(data, W_fixed, r, H_max; method=:multiplicativo, max_iter=60)
    cols = size(data, 2)
    H_init = rand_pos(r, cols) 

    if method == :multiplicativo
        H_proj, _ = multiplicative_H_projection(
            data,
            W_fixed,
            H_init;
            max_iter=max_iter,
            tol=1e-4
        )

        return H_proj
    elseif method == :lin
        H_proj, _, _ = projected_gradient_lin_H(data, W_fixed, H_init, H_max; 
                                                alpha_init=1.0, tol=1e-4, max_iter=max_iter)
        return H_proj
    end
end

function main()
    models = Dict{Symbol, Function}(
        :multiplicativo => nmf_multiplicative,
        :lin            => nmf_lin_algorithm,
    )

    println("--- Carregando Dataset ---")
    train_matrix = []
    train_labels = []
    test_matrix = []
    test_labels = []

    for person_id in 1:40
        folder_path = joinpath(DATA_PATH, "s$person_id")
        if !isdir(folder_path); continue; end
        images_files = sort([joinpath(folder_path, f) for f in readdir(folder_path) if endswith(f, ".pgm")])

        for (img_idx, img_path) in enumerate(images_files)
            img = Float64.(Gray.(load(img_path)))
            img_vec = vec(img)
            if img_idx <= NUM_TRAIN_PER_PERSON
                push!(train_matrix, img_vec)
                push!(train_labels, person_id)
            else
                push!(test_matrix, img_vec)
                push!(test_labels, person_id)
            end
        end
    end
    X_train = hcat(train_matrix...)
    X_test = hcat(test_matrix...)
    m, n_train = size(X_train)
    _, n_test = size(X_test)
    println("Dados carregados: $n_train Treino, $n_test Teste")

    # =========================================================================
    # Loop de Execução
    # =========================================================================

    results_summary = []

    for rank in RANKS
        println("\n========================================")
        println("EXECUTANDO COM RANK = $rank")
        println("========================================")

        Random.seed!(1234)
        W_init_common = rand_pos(m, rank)
        H_init_common = rand_pos(rank, n_train)

        for (model_sym, algo_func) in models
            model_name = string(model_sym)
            println("\n>>> Preparando Modelo: $model_name (rank=$rank)")

            Random.seed!(1234)

            OUTPUT_DIR = joinpath("resultados", "face_recognition", "$(model_name)_Rank$(rank)")
            if !isdir(OUTPUT_DIR); mkpath(OUTPUT_DIR); end
            log_path = joinpath(OUTPUT_DIR, "execution.log")

            open(log_path, "w") do io
                log_msg(io, "SESSION_START: Face Recognition Experiment")
                log_msg(io, "SETUP: Model=$model_name | Rank=$rank | MaxIter=$MAX_ITER")
                log_msg(io, "STATUS: Starting Training Loop...")
                
                ext_iters = 0
                int_iters = 0

                if model_sym == :lin
                    W_train, H_train, errors, t_train, ext_iters, int_iters, _, _, _ = algo_func(
                        X_train, rank,
                        copy(W_init_common), copy(H_init_common);
                        max_iter=MAX_ITER, tol=TOL,
                        log_io=io, log_interval=10 
                    )
                else
                    W_train, H_train, errors, t_train, iters = algo_func(
                        X_train, rank,
                        copy(W_init_common), copy(H_init_common);
                        max_iter=MAX_ITER, tol=TOL,
                        log_io=io, log_interval=10 
                    )
                    ext_iters = iters
                    int_iters = 0   # multiplicativo não tem iterações internas
                end

                println(io, "")
                log_msg(io, "STATUS: Training Finished. Time=$(round(t_train, digits=4))s")
                log_msg(io, "STATUS: Iterations - Ext: $ext_iters | Int: $int_iters")
                log_msg(io, "STATUS: Projecting Test Data and Classifying...")

                H_test = project_new_data(X_test, W_train, rank, 1e6; method=model_sym)  

                println(io, "")
                println(io, "=== CLASSIFICATION REPORT ===")
                println(io, "IDX | REAL_ID | PRED_ID | DISTANCE | MATCH_IDX | STATUS")
                println(io, "--------------------------------------------------------")

                acertos = 0
                for i in 1:n_test
                    h_unk = H_test[:, i]
                    real_id = test_labels[i]
                    
                    min_dist = Inf
                    predicted_id = -1
                    match_idx = -1
                    for j in 1:n_train
                        dist = norm(h_unk - H_train[:, j])
                        if dist < min_dist
                            min_dist = dist
                            predicted_id = train_labels[j]
                            match_idx = j
                        end
                    end
                    
                    is_correct = (predicted_id == real_id)
                    if is_correct; acertos += 1; end
                    
                    status_str = is_correct ? "HIT " : "MISS"
                    
                    @printf(io, "%03d |   %02d    |   %02d    |  %.4f  |   %04d    | %s\n", 
                            i, real_id, predicted_id, min_dist, match_idx, status_str)
                end

                acc = (acertos / n_test) * 100
                
                println(io, "--------------------------------------------------------")
                log_msg(io, "SUMMARY: Accuracy=$(round(acc, digits=2))% ($acertos/$n_test)")
                log_msg(io, "SESSION_END")

                println("   -> Modelo: $model_name (rank=$rank) | Acurácia: $(round(acc, digits=2))% | Ext: $ext_iters | Int: $int_iters")
                push!(results_summary, (model_name, rank, acc, t_train, ext_iters, int_iters))
            end
        end
    end

    # =========================================================================
    # Resumo final
    # =========================================================================

    println("\n========================================")
    println("RESUMO FINAL")
    println("========================================")
    @printf "%-15s | %-4s | %-9s | %-9s | %-10s | %-6s\n" "Modelo" "Rank" "Iter Ext" "Iter Int" "Tempo (s)" "Acc (%)"
    println("-"^70)

    sort!(results_summary, by=x -> (x[2], x[1]))

    for (name, rank, acc, time_s, ext_iters, int_iters) in results_summary
        @printf "%-15s | %-4d | %-9d | %-9d | %-10.2f | %-6.2f\n" name rank ext_iters int_iters time_s acc
    end
end

main()