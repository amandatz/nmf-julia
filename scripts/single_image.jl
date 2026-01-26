using Pkg; Pkg.activate(".")
using Images, FileIO
using LinearAlgebra
using Plots
using Statistics
using Printf

try
    using Revise
catch
end

includet(joinpath(@__DIR__, "../src/NMFProject.jl")) 
using .NMFProject

function main()
    target_img_path = "data/att_face_dataset/s1/1.pgm" 
    rank_val        = 20
    max_iter        = 500
    tolerance       = 1e-5

    Random.seed!(1234)

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

        :pca => NMFProject.nmf_pca_wrapper,
    )

    # --- CARREGAMENTO ---
    println("--- Carregando imagem alvo: $target_img_path ---")
    if !isfile(target_img_path)
        error("Arquivo não encontrado! Verifique o caminho: $(abspath(target_img_path))")
    end

    img_raw = load(target_img_path)
    X = Float64.(Gray.(img_raw))
    m, n = size(X)
    println("Matriz X: $m x $n | Rank: $rank_val")

    # --- EXECUÇÃO ---
    println("Gerando W e H iniciais (comuns a todos os modelos)...")
    W_init_common = rand(m, rank_val)
    H_init_common = rand(rank_val, n)

    for (model_sym, algo_func) in models
        model_name = string(model_sym)
        
        println("\n========================================")
        println("   Rodando: $model_name")
        println("========================================")
        
        W, H, errors, t_total, iters = algo_func(
            X, rank_val, 
            copy(W_init_common), copy(H_init_common); 
            max_iter=max_iter, tol=tolerance
        )
        
        # Prints de resultado
        println("\n       Tempo: $(round(t_total, digits=3))s")
        println("       Iterações: $iters")
        err_final = isempty(errors) ? NaN : last(errors)
        println("       Erro Final: $(round(err_final, digits=5))")

        # Salvamento
        output_dir = joinpath("resultados", "single_image", model_name)
        if !isdir(output_dir)
            mkpath(output_dir)
        end
        
        # Gráfico de convergência
        p_conv = plot(errors, title="Convergência - $model_name", 
                      xlabel="Iterações", ylabel="Erro Frobenius", yscale=:log10, 
                      legend=false, lc=:blue, lw=2)
        savefig(p_conv, joinpath(output_dir, "01_convergencia.png"))

        # Reconstrução visual
        X_recon = W * H
        erro_rel = norm(X - X_recon) / norm(X)
        erro_perc = round(erro_rel * 100, digits=2)

        p_comp = plot(
            heatmap(X, c=:grays, yflip=true, title="Original", axis=false, legend=false, aspect_ratio=:equal),
            heatmap(X_recon, c=:grays, yflip=true, title="Reconstrução ($model_name)\nErro: $erro_perc%", axis=false, legend=false, aspect_ratio=:equal),
            layout=(1, 2), size=(800, 450)
        )
        savefig(p_comp, joinpath(output_dir, "02_Comparacao.png"))
        
        println("       Resultados salvos em: $output_dir")
    end

    println("\n=== Modelos finalizados ===")
end

main()