function nmf_pca_wrapper(X, r, W_init, H_init; kwargs...)
    t_start = time()
    
    # 1. Decomposição SVD: X = U * S * V'
    F = svd(X) 
    
    # 2. Truncamento para o Rank r (Melhor aproximação possível)
    # W será os vetores singulares à esquerda (Eigenfaces)
    # H será a mistura de valores singulares e vetores à direita
    W = F.U[:, 1:r]
    H = Diagonal(F.S[1:r]) * F.Vt[1:r, :] 
    
    t_total = time() - t_start
    
    # 3. Cálculo do erro (apenas para registro)
    final_error = norm(X - W*H)
    
    # Retorna vetor de erros constante (não iterativo) e 1 iteração
    errors = [final_error]
    iters = 1 
    
    return W, H, errors, t_total, iters
end