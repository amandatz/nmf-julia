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
const RANK      = 40
const MAX_ITER  = 300
const TOL       = 1e-4
const NUM_TRAIN_PER_PERSON = 7 
const IMG_SIZE = (112, 92)

function log_msg(io::IO, msg::String)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(io, "[$t] $msg")
    println("[$t] $msg") 
end

function project_new_data(data, W_fixed, r; method=:multiplicativo, max_iter=60)
    cols = size(data, 2)
    H_init = rand(r, cols) 

    if method == :multiplicativo
        H_proj = copy(H_init)
        WtV = W_fixed' * data
        WtW = W_fixed' * W_fixed
        for i in 1:max_iter
            H_proj .= H_proj .* (WtV ./ (WtW * H_proj .+ 1e-9))
        end
        return H_proj

    elseif method == :lin
        H_proj, _, _ = projected_gradient_lin_H(data, W_fixed, H_init; 
                                                alpha_init=1.0, tol=1e-4, max_iter=max_iter)
        return H_proj
    end
end

function main()
    models = [
        :lin => nmf_lin_algorithm, 
        :multiplicativo => NMFProject.nmf_multiplicative
    ]

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
    
    Random.seed!(1234)
    W_init_common = rand(m, RANK)
    H_init_common = rand(RANK, n_train)

    results_summary = []

    for (model_sym, algo_func) in models
        model_name = string(model_sym)
        println("\n>>> Preparando Modelo: $model_name")

        Random.seed!(1234) # reset da seed

        OUTPUT_DIR = joinpath("resultados", "face_recognition", "$(model_name)_Rank$(RANK)")
        if !isdir(OUTPUT_DIR); mkpath(OUTPUT_DIR); end
        log_path = joinpath(OUTPUT_DIR, "execution.log")

        open(log_path, "w") do io
            log_msg(io, "SESSION_START: Face Recognition Experiment")
            log_msg(io, "SETUP: Model=$model_name | Rank=$RANK | MaxIter=$MAX_ITER")
            log_msg(io, "STATUS: Starting Training Loop...")
            
            W_train, H_train, errors, t_train, iters = algo_func(
                X_train, RANK,
                copy(W_init_common), copy(H_init_common);
                max_iter=MAX_ITER, tol=TOL,
                log_io=io, log_interval=20 
            )

            println(io, "")
            log_msg(io, "STATUS: Training Finished. Time=$(round(t_train, digits=4))s")
            log_msg(io, "STATUS: Projecting Test Data and Classifying...")
            
            H_test = project_new_data(X_test, W_train, RANK; method=model_sym)

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

            println("   -> Modelo: $model_name | Acurácia: $(round(acc, digits=2))% | Log gerado.")
            push!(results_summary, (model_name, acc, t_train))
        end
    end

    println("\n========================================")
    println("RESUMO FINAL")
    println("========================================")
    sort!(results_summary, by=x -> x[2], rev=true)
    for (name, acc, time_s) in results_summary
        @printf "%-25s | Acurácia: %6.2f%% | Tempo Treino: %6.2fs\n" name acc time_s
    end
end

main()