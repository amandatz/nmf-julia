function nmf_pca_wrapper(X, r, W_init, H_init; kwargs...)
    t_start = time()
    
    F = svd(X) 
    
    # Truncamento para o rank r
    W = F.U[:, 1:r]
    H = Diagonal(F.S[1:r]) * F.Vt[1:r, :] 
    
    t_total = time() - t_start
    
    final_error = norm(X - W*H)
    
    errors = [final_error]
    iters = 1 
    
    return W, H, errors, t_total, iters
end