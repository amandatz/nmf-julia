function nmf_pca_wrapper(
    X, r, W_init, H_init; 
    log_io::IO = stdout, 
    log_interval::Int = 10, 
    kwargs...
)
    t_start = time()
    
    # --- CABEÇALHO DO LOG ---
    timestamp = Dates.format(now(), "HH:MM:SS")
    println(log_io, "[$timestamp] [PCA_ALGO]  Starting Direct SVD Decomposition (Rank=$r)")
    println(log_io, "--------------------------------------------------------------------------")

    # Log de status
    println(log_io, "[$timestamp] [PCA_ALGO]  Computing Singular Value Decomposition...")

    F = svd(X) 
    
    # Truncamento para o rank r
    W = F.U[:, 1:r]
    H = Diagonal(F.S[1:r]) * F.Vt[1:r, :] 
    
    t_total = time() - t_start
    
    final_error = norm(X - W*H)
    errors = [final_error]
    iters = 1 
    
    # --- LOG DE RESULTADO ---
    t_now = Dates.format(now(), "HH:MM:SS")
    @printf(log_io, "[%s] [PCA_ALGO]  DONE | Final Error: %.6e | Time: %.4fs\n", 
            t_now, final_error, t_total)
    
    println(log_io, "--------------------------------------------------------------------------")

    return W, H, errors, t_total, iters
end