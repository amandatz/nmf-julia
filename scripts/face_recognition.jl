using Pkg;
Pkg.activate(".");
using Images, FileIO
using LinearAlgebra
using Statistics
using Printf
using Random
using Plots

try
    using Revise
catch
end
includet("../src/NMFProject.jl")
using .NMFProject

# =========================================================================
# CONFIGURAÇÕES GERAIS
# =========================================================================

const DATA_PATH = "data/att_face_dataset"
const RANK      = 40
const MAX_ITER  = 300
const TOL       = 1e-4
const NUM_TRAIN_PER_PERSON = 7  # O dataset tem 10 imagens por pessoa, usamos 7 para treino e 3 para teste
const IMG_SIZE = (112, 92)

function project_new_data(data, W_fixed, r)
    cols = size(data, 2)
    H_proj = rand(r, cols)
    WtV = W_fixed' * data
    WtW = W_fixed' * W_fixed
    for i in 1:60
        H_proj .= H_proj .* (WtV ./ (WtW * H_proj .+ 1e-9))
    end
    return H_proj
end

function main()
    models = Dict{Symbol,Function}(
        # :multiplicativo => NMFProject.nmf_multiplicative,
        :lin => NMFProject.nmf_lin_algorithm,
        # :pg_spectral_non_monotone => (X, r, W, H; kwargs...) -> NMFProject.nmf_gradient_projected(
        #     X, r, W, H;
        #     alpha_rule_W=NMFProject.make_rule_spectral_W(),
        #     alpha_rule_H=NMFProject.make_rule_spectral_H(),
        #     monotone=false,
        #     kwargs...
        # ),
        # :pca => NMFProject.nmf_pca_wrapper,
    )

    println("--- Separando Dataset (70% Treino / 30% Teste) ---")

    train_matrix = []
    train_labels = []
    test_matrix = []
    test_labels = []

    for person_id in 1:40
        folder_path = joinpath(DATA_PATH, "s$person_id")
        # Ordena para garantir reprodutibilidade
        images_files = sort([joinpath(folder_path, f) for f in readdir(folder_path) if endswith(f, ".pgm")])

        for (img_idx, img_path) in enumerate(images_files)
            img = Float64.(Gray.(load(img_path)))
            img_vec = vec(img)

            if img_idx <= NUM_TRAIN_PER_PERSON      # TREINO
                push!(train_matrix, img_vec)
                push!(train_labels, person_id)
            else                                    # TESTE
                push!(test_matrix, img_vec)
                push!(test_labels, person_id)
            end
        end
    end

    X_train = hcat(train_matrix...)
    X_test = hcat(test_matrix...)
    m, n_train = size(X_train)
    _, n_test = size(X_test)
    IMG_SIZE = (112, 92)

    println("Total Treino: $n_train imagens (7 por pessoa)")
    println("Total Teste : $n_test imagens (3 por pessoa)")

    # =========================================================================
    # LOOP DE COMPARAÇÃO
    # =========================================================================
    
    println("Gerando condições iniciais...")
    W_init_common = rand(m, RANK)
    H_init_common = rand(RANK, n_train)

    results_summary = []

    for (model_sym, algo_func) in models
        model_name = string(model_sym)
        println("\n>>> Modelo: $model_name")

        W_train, H_train, errors, t_train, iters = algo_func(
            X_train, RANK,
            copy(W_init_common), copy(H_init_common);
            max_iter=MAX_ITER, tol=TOL
        )

        H_test = project_new_data(X_test, W_train, RANK)

        acertos = 0
        OUTPUT_DIR = joinpath("resultados", "face_recognition", "$(model_name)_Rank$(RANK)")
        if !isdir(OUTPUT_DIR)
            mkpath(OUTPUT_DIR)
        end

        open(joinpath(OUTPUT_DIR, "log_detalhado.txt"), "w") do io
            println(io, "RELATÓRIO DE EXECUÇÃO: $model_name")
            println(io, "-----------------------------------")

            for i in 1:n_test
                h_unk = H_test[:, i]
                real_id = test_labels[i]

                # Nearest Neighbor
                min_dist = Inf
                predicted_id = -1

                for j in 1:n_train
                    dist = norm(h_unk - H_train[:, j])
                    if dist < min_dist
                        min_dist = dist
                        predicted_id = train_labels[j]
                    end
                end

                if predicted_id == real_id
                    acertos += 1
                else
                    println(io, "ERRO: Teste $i (Real: $real_id) -> Predito: $predicted_id")
                end
            end

            acc = (acertos / n_test) * 100
            println(io, "\n===================================")
            println(io, "TOTAL: $acertos acertos em $n_test testes.")
            println(io, "ACURÁCIA: $(round(acc, digits=2))%")

            println(io, "TEMPO DE TREINO: $(round(t_train, digits=4))s")

            push!(results_summary, (model_name, acc, t_train))
            println("       Acurácia: $(round(acc, digits=2))% | Tempo: $(round(t_train, digits=2))s")
        end
    end

    println("\n========================================")
    println("RESUMO FINAL (SPLIT 70/30)")
    println("========================================")
    sort!(results_summary, by=x -> x[2], rev=true)

    for (name, acc, time_s) in results_summary
        @printf "%-25s | Acurácia: %6.2f%% | Tempo Treino: %6.2fs\n" name acc time_s
    end
end

main()