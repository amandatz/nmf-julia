using LinearAlgebra
using Statistics
using Printf
using Dates

# ==========================================================
# Subproblemas (W e H)
# ==========================================================

function projected_gradient_W(X, H, W0; alpha_init=1e-3, lambda=0.0, tol=1e-6, max_iter=1000, monotone=true, alpha_rule_W=(args...)->args[5])
    W = copy(W0)
    alpha = alpha_init
    f_W(Wt, Ht) = 0.5 * norm(X - Wt*Ht)^2 + (lambda/2)*norm(Wt)^2
    f_hist_W = nothing

    for iter = 1:max_iter
        W_old = copy(W)
        GW = W * (H * H') .- X * H' .+ lambda .* W
        alpha = alpha_rule_W(W, H, GW, iter, alpha)
        W .= max.(W .- alpha .* GW, 0.0)

        # Line Search
        W, _, f_hist_W = line_search_segment!(W_old, W, GW, H, f_W; monotone=monotone, f_hist=f_hist_W)
        
        if norm(W - W_old)/max(1.0, norm(W_old)) < tol; return W, iter; end
    end
    return W, max_iter
end

function projected_gradient_H(X, W, H0; alpha_init=1e-3, lambda=0.0, tol=1e-6, max_iter=1000, monotone=true, alpha_rule_H=(args...)->args[5])
    H = copy(H0)
    alpha = alpha_init
    f_H(Ht, Wt) = 0.5 * norm(X - Wt*Ht)^2 + (lambda/2)*norm(Ht)^2
    f_hist_H = nothing

    for iter = 1:max_iter
        H_old = copy(H)
        GH = (W' * W) * H .- W' * X .+ lambda .* H
        alpha = alpha_rule_H(W, H, GH, iter, alpha)
        H .= max.(H .- alpha .* GH, 0.0)

        # Line Search
        H, _, f_hist_H = line_search_segment!(H_old, H, GH, W, f_H; monotone=monotone, f_hist=f_hist_H)

        if norm(H - H_old)/max(1.0, norm(H_old)) < tol; return H, iter; end
    end
    return H, max_iter
end

# ==========================================================
# Função Principal
# ==========================================================

function nmf_gradient_projected(X, r, W_init, H_init; 
                                max_iter=200, 
                                tol=1e-4, 
                                sub_tol=1e-3, 
                                sub_max_iter=200, 
                                monotone=true, 
                                alpha_rule_W=(args...)->args[5], 
                                alpha_rule_H=(args...)->args[5],
                                alpha_init=1e-3,
                                lambda=0.0,
                                log_io::IO = stdout,    # Onde escrever o log
                                log_interval::Int = 10, # Intervalo de escrita
                                kwargs...)
    
    W = copy(W_init); H = copy(H_init)
    errors = Float64[]
    t_start = time()
    total_sub = 0

    # --- CABEÇALHO DO LOG ---
    timestamp = Dates.format(now(), "HH:MM:SS")
    println(log_io, "[$timestamp] [PG_ALGO] Starting Projected Gradient Optimization")
    println(log_io, "[$timestamp] [PG_ALGO] Config: MaxIter=$max_iter | Tol=$tol | Monotone=$monotone | Lambda=$lambda")
    println(log_io, "[$timestamp] [PG_ALGO] SubConfig: SubMaxIter=$sub_max_iter | SubTol=$sub_tol")
    println(log_io, "[$timestamp] [PG_ALGO] ITER |  RECON_ERROR  |   DELTA_W  |   DELTA_H  | SUB_W | SUB_H")
    println(log_io, "--------------------------------------------------------------------------------------")

    converged = false

    for iter = 1:max_iter
        
        W_old = copy(W); H_old = copy(H)

        # Subproblema W
        W, iW = projected_gradient_W(X, H, W; 
            tol=sub_tol, max_iter=sub_max_iter, monotone=monotone, 
            alpha_rule_W=alpha_rule_W, alpha_init=alpha_init, lambda=lambda)
        total_sub += iW
        
        # Subproblema H
        H, iH = projected_gradient_H(X, W, H; 
            tol=sub_tol, max_iter=sub_max_iter, monotone=monotone, 
            alpha_rule_H=alpha_rule_H, alpha_init=alpha_init, lambda=lambda)
        total_sub += iH

        # Métricas
        current_error = norm(X - W * H)
        push!(errors, current_error)

        deltaW = norm(W - W_old)/max(1, norm(W_old))
        deltaH = norm(H - H_old)/max(1, norm(H_old))

        # --- Log Periódico ---
        if iter == 1 || iter % log_interval == 0
            t_now = Dates.format(now(), "HH:MM:SS")
            @printf(log_io, "[%s] [PG_ALGO] %04d | %.6e | %.4e | %.4e |  %03d  |  %03d\n", 
                    t_now, iter, current_error, deltaW, deltaH, iW, iH)
            flush(log_io)
        end

        # Critério de Parada
        if deltaW < tol && deltaH < tol
            converged = true
            t_now = Dates.format(now(), "HH:MM:SS")
            println(log_io, "[$t_now] [PG_ALGO] CONVERGED at Iter $iter (DeltaW=$deltaW, DeltaH=$deltaH)")
            break
        end
    end

    if !converged
        t_now = Dates.format(now(), "HH:MM:SS")
        println(log_io, "[$t_now] [PG_ALGO] STOPPED: Max Iterations Reached ($max_iter)")
    end

    elapsed = time() - t_start
    println(log_io, "--------------------------------------------------------------------------------------")

    return W, H, errors, elapsed, total_sub
end