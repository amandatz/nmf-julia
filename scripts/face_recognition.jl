using Pkg; Pkg.activate(".")
using Images, FileIO
using LinearAlgebra
using Statistics
using Printf
using Random
using Plots

# Carrega o módulo (com Revise)
try
    using Revise
catch
end
includet("../src/NMFProject.jl")
using .NMFProject

# =========================================================================
# 1. CONFIGURAÇÕES GERAIS
# =========================================================================
DATA_PATH = "data/att_face_dataset"
RANK      = 40      # Número de features (pode testar 20, 40, 60)
MAX_ITER  = 300     # Iterações de treino
TOL       = 1e-4

# Usar a 10ª foto de cada pessoa para teste (deixando 9 para treino)
TEST_IMAGE_INDEX = 10 

# --- DICIONÁRIO DE MODELOS ---
models = Dict{Symbol, Function}(
    :multiplicativo => NMFProject.nmf_multiplicative,
    
    :lin => NMFProject.nmf_lin_algorithm,

    :pg_spectral_non_monotone => (X, r, W, H; kwargs...) -> NMFProject.nmf_gradient_projected(
        X, r, W, H;
        alpha_rule_W = NMFProject.make_rule_spectral_W(),
        alpha_rule_H = NMFProject.make_rule_spectral_H(),
        monotone = false,
        kwargs...
    ),

    :pg_spectral_monotone => (X, r, W, H; kwargs...) -> NMFProject.nmf_gradient_projected(
        X, r, W, H;
        alpha_rule_W = NMFProject.make_rule_spectral_W(),
        alpha_rule_H = NMFProject.make_rule_spectral_H(),
        monotone = true,
        kwargs...
    ),
)

# =========================================================================
# 2. CARREGAMENTO E SPLIT (IGUAL PARA TODOS)
# =========================================================================
println("--- Carregando e separando Dataset... ---")

train_matrix = []
train_labels = []
test_matrix  = []
test_labels  = []

for person_id in 1:40
    folder_path = joinpath(DATA_PATH, "s$person_id")
    images_files = sort([joinpath(folder_path, f) for f in readdir(folder_path) if endswith(f, ".pgm")])
    
    for (img_idx, img_path) in enumerate(images_files)
        img = Float64.(Gray.(load(img_path)))
        img_vec = vec(img)
        
        if img_idx == TEST_IMAGE_INDEX
            push!(test_matrix, img_vec)
            push!(test_labels, person_id)
        else
            push!(train_matrix, img_vec)
            push!(train_labels, person_id)
        end
    end
end

X_train = hcat(train_matrix...)
X_test  = hcat(test_matrix...)
m, n_train = size(X_train)
_, n_test  = size(X_test)
IMG_SIZE = (112, 92)

println("Dados prontos. Treino: $n_train | Teste: $n_test")

# =========================================================================
# 3. FUNÇÃO DE PROJEÇÃO (INFERÊNCIA)
# =========================================================================
# Para ser justo, usaremos o mesmo método para projetar os dados de teste em W
# (Encontrar H_test tal que X_test ≈ W_train * H_test)
function project_new_data(data, W_fixed, r)
    cols = size(data, 2)
    H_proj = rand(r, cols)
    WtV = W_fixed' * data
    WtW = W_fixed' * W_fixed
    # 60 iterações de Regra Multiplicativa (padrão robusto para inferência)
    for i in 1:60
        H_proj .= H_proj .* (WtV ./ (WtW * H_proj .+ 1e-9))
    end
    return H_proj
end

# =========================================================================
# 4. LOOP DE COMPARAÇÃO
# =========================================================================

# Condições iniciais comuns para o TREINO (Fair Play)
println("Gerando condições iniciais aleatórias fixas...")
W_init_common = rand(m, RANK)
H_init_common = rand(RANK, n_train)

# Tabela para guardar resumo final
results_summary = []

for (model_sym, algo_func) in models
    model_name = string(model_sym)
    println("\n>>> Iniciando Modelo: $model_name")
    
    # 1. TREINO
    t_start = time()
    # Copia os inits para não afetar o próximo loop
    W_train, H_train, errors, t_train, iters = algo_func(
        X_train, RANK, 
        copy(W_init_common), copy(H_init_common); 
        max_iter=MAX_ITER, tol=TOL
    )
    
    # 2. PROJEÇÃO (TESTE)
    H_test = project_new_data(X_test, W_train, RANK)
    
    # 3. CLASSIFICAÇÃO
    acertos = 0
    OUTPUT_DIR = joinpath("resultados_reconhecimento", "$(model_name)_Rank$(RANK)")
    if !isdir(OUTPUT_DIR) mkpath(OUTPUT_DIR) end
    dir_acertos = joinpath(OUTPUT_DIR, "acertos")
    dir_erros   = joinpath(OUTPUT_DIR, "erros")
    mkpath(dir_acertos); mkpath(dir_erros)
    
    open(joinpath(OUTPUT_DIR, "relatorio.txt"), "w") do io
        println(io, "MODELO: $model_name")
        println(io, "Tempo de Treino: $(round(t_train, digits=2))s")
        println(io, "Erro Final (Treino): $(last(errors))\n")
        
        for i in 1:n_test
            h_unk = H_test[:, i]
            real_id = test_labels[i]
            
            # Nearest Neighbor
            min_dist = Inf
            predicted_id = -1
            best_match_idx = -1
            
            for j in 1:n_train
                dist = norm(h_unk - H_train[:, j])
                if dist < min_dist
                    min_dist = dist
                    predicted_id = train_labels[j]
                    best_match_idx = j
                end
            end
            
            is_correct = (predicted_id == real_id)
            if is_correct acertos += 1 end
            
            status = is_correct ? "ACERTOU" : "ERROU"
            println(io, "Img $i (Real: $real_id) -> Pred: $predicted_id [$status]")
            
            # Salvar imagem comparativa
            img_test_px  = reshape(X_test[:, i], IMG_SIZE)
            img_match_px = reshape(X_train[:, best_match_idx], IMG_SIZE)
            
            p = plot(
                heatmap(img_test_px, c=:grays, yflip=true, axis=false, title="Teste ($real_id)"),
                heatmap(img_match_px, c=:grays, yflip=true, axis=false, title="Predito ($predicted_id)"),
                layout=(1, 2), size=(600, 300),
                plot_title="$status ($model_name)",
                plot_titlefont=font(10, is_correct ? :green : :red)
            )
            
            fname = "teste$(i)_real$(real_id)_pred$(predicted_id).png"
            savefig(p, joinpath(is_correct ? dir_acertos : dir_erros, fname))
        end
        
        acc = (acertos / n_test) * 100
        println(io, "\nACURÁCIA: $(round(acc, digits=2))%")
        
        # Guarda para o resumo geral
        push!(results_summary, (model_name, acc, t_train))
        println("   ✅ Finalizado $model_name: Acurácia $(round(acc, digits=2))%")
    end
end

println("\n========================================")
println("RESUMO FINAL DA COMPARAÇÃO")
println("========================================")
# Ordena por Acurácia (maior para menor)
sort!(results_summary, by = x -> x[2], rev=true)

for (name, acc, time_s) in results_summary
    @printf "%-25s | Acurácia: %6.2f%% | Tempo Treino: %6.2fs\n" name acc time_s
end