using Pkg; Pkg.activate(".")
using Images, FileIO
using LinearAlgebra
using Plots
using Statistics
using Printf

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
TARGET_IMG_PATH = "data/att_face_dataset/s1/1.pgm" 
RANK            = 20    # Tente variar: 10, 20, 40
MAX_ITER        = 500
TOLERANCE       = 1e-5

# =========================================================================
# 2. DEFINIÇÃO DOS MODELOS (SEU DICIONÁRIO)
# =========================================================================
# Nota: Adicionei o prefixo 'NMFProject.' para garantir que o Julia encontre 
# as funções dentro do módulo.

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
# 3. CARREGAMENTO DA IMAGEM
# =========================================================================
println("--- Carregando imagem alvo: $TARGET_IMG_PATH ---")
if !isfile(TARGET_IMG_PATH)
    error("Arquivo não encontrado! Verifique o caminho.")
end

img_raw = load(TARGET_IMG_PATH)
X = Float64.(Gray.(img_raw)) # X é a própria imagem (não vetorizada)
m, n = size(X)
println("Matriz X: $m x $n | Rank: $RANK")

# =========================================================================
# 4. LOOP DE EXECUÇÃO
# =========================================================================

# Condição inicial idêntica para todos (para comparação justa)
println("Gerando W e H iniciais (comuns a todos os modelos)...")
W_init_common = rand(m, RANK)
H_init_common = rand(RANK, n)

for (model_sym, algo_func) in models
    model_name = string(model_sym) # Converte :lin para "lin"
    
    println("\n========================================")
    println("▶️ Rodando: $model_name")
    println("========================================")
    
    # Roda o algoritmo
    t_start = time()
    
    # IMPORTANTE: Usamos copy() para não sujar as matrizes iniciais
    # Passamos max_iter e tol via kwargs... que suas funções anônimas aceitam
    W, H, errors, t_total, iters = algo_func(
        X, RANK, 
        copy(W_init_common), copy(H_init_common); 
        max_iter=MAX_ITER, tol=TOLERANCE
    )
    
    println("   ↳ Tempo: $(round(t_total, digits=3))s")
    println("   ↳ Iterações: $iters")
    println("   ↳ Erro Final: $(round(last(errors), digits=5))")

    # --- SALVAMENTO ---
    output_dir = joinpath("resultados", "single_image", model_name)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # 1. Gráfico de Convergência
    p_conv = plot(errors, title="Convergência - $model_name", 
                  xlabel="Iterações", ylabel="Erro Frobenius", yscale=:log10, 
                  legend=false, lc=:blue, lw=2)
    savefig(p_conv, joinpath(output_dir, "01_convergencia.png"))

    # 2. Reconstrução Visual e Erro
    X_recon = W * H
    erro_rel = norm(X - X_recon) / norm(X)
    erro_perc = round(erro_rel * 100, digits=2)

    p_comp = plot(
        heatmap(X, c=:grays, yflip=true, title="Original", axis=false, legend=false, aspect_ratio=:equal),
        heatmap(X_recon, c=:grays, yflip=true, title="Reconstrução ($model_name)\nErro: $erro_perc%", axis=false, legend=false, aspect_ratio=:equal),
        layout=(1, 2), size=(800, 450)
    )
    savefig(p_comp, joinpath(output_dir, "02_Comparacao.png"))
    
    println("   ✅ Resultados salvos em: $output_dir")
end

println("\n🏁 Todos os modelos finalizados!")