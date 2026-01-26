using Pkg
Pkg.activate(".")
using Images, FileIO
using LinearAlgebra
using Statistics
using Printf
using Random
using Plots
using Dates

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
const MAX_ITER  = 500
const TOL       = 1e-4
const NUM_TRAIN_PER_PERSON = 7 
const IMG_SIZE = (112, 92)

# Função auxiliar para log com timestamp
function log_msg(io::IO, msg::String)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(io, "[$t] $msg")
    # Se quiser ver no terminal ao mesmo tempo, descomente abaixo:
    println("[$t] $msg") 
end

# Função de projeção para dados de teste (mantida igual)
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
        :multiplicativo => NMFProject.nmf_multiplicative,
        :lin => NMFProject.nmf_lin_algorithm,
        :pg_spectral_non_monotone => (X, r, W, H; kwargs...) -> NMFProject.nmf_gradient_projected(
            X, r, W, H;
            alpha_rule_W=NMFProject.make_rule_spectral_W(),
            alpha_rule_H=NMFProject.make_rule_spectral_H(),
            monotone=false,
            kwargs... 
        ),
        :pca => NMFProject.nmf_pca_wrapper,
    )

    # =========================================================================
    # SPLIT (CARREGAMENTO DOS DADOS)
    # =========================================================================
    
    println("--- Separando Dataset (70% Treino / 30% Teste) ---")

    train_matrix = []
    train_labels = []
    test_matrix = []
    test_labels = []

    for person_id in 1:40
        folder_path = joinpath(DATA_PATH, "s$person_id")
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
    
    println("Total Treino: $n_train imagens")
    println("Total Teste : $n_test imagens")

    # =========================================================================
    # LOOP DE COMPARAÇÃO
    # =========================================================================
    
    println("Gerando condições iniciais...")
    W_init_common = rand(m, RANK)
    H_init_common = rand(RANK, n_train)

    results_summary = []

    for (model_sym, algo_func) in models
        model_name = string(model_sym)
        println("\n>>> Preparando Modelo: $model_name")

        # 1. CRIAR DIRETÓRIO E ARQUIVO ANTES DO TREINO
        OUTPUT_DIR = joinpath("resultados", "face_recognition", "$(model_name)_Rank$(RANK)")
        if !isdir(OUTPUT_DIR)
            mkpath(OUTPUT_DIR)
        end
        log_path = joinpath(OUTPUT_DIR, "execution.log")

        # Abre o arquivo agora para capturar TUDO
        open(log_path, "w") do io
            
            # --- Cabeçalho Geral ---
            log_msg(io, "SESSION_START: Face Recognition Experiment")
            log_msg(io, "SETUP: Model=$model_name | Rank=$RANK | MaxIter=$MAX_ITER")

            # 2. EXECUTA O ALGORITMO PASSANDO O IO
            # Agora passamos 'io' para dentro da função. 
            # O algoritmo vai escrever o progresso iterativo neste mesmo arquivo.
            log_msg(io, "STATUS: Starting Training Loop...")
            println(io, "") # Espaçamento
            
            W_train, H_train, errors, t_train, iters = algo_func(
                X_train, RANK,
                copy(W_init_common), copy(H_init_common);
                max_iter=MAX_ITER, tol=TOL,
                log_io=io,          # <--- O PULO DO GATO: O log interno vai aqui
                log_interval=20     # <--- Define a frequência do log interno
            )

            println(io, "") # Espaçamento pós-treino
            log_msg(io, "STATUS: Training Finished. Time=$(round(t_train, digits=4))s")

            # 3. PROJEÇÃO E TESTE
            log_msg(io, "STATUS: Projecting Test Data and Classifying...")
            
            H_test = project_new_data(X_test, W_train, RANK)

            println(io, "")
            println(io, "=== CLASSIFICATION REPORT ===")
            println(io, "IDX | REAL_ID | PRED_ID | DISTANCE | MATCH_IDX | STATUS")
            println(io, "--------------------------------------------------------")

            acertos = 0
            
            for i in 1:n_test
                h_unk = H_test[:, i]
                real_id = test_labels[i]

                # Nearest Neighbor
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

            # Feedback no console (Terminal)
            println("   -> Concluído! Acurácia: $(round(acc, digits=2))% | Log: $log_path")
            push!(results_summary, (model_name, acc, t_train))
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