# =========================================================================
# Funções de Otimização
# =========================================================================

function projected_gradient_lin_W(X, H, W0; alpha_init = 1.0, tol = 1e-4, max_iter = 50)
    W = copy(W0)
    alpha = alpha_init
    beta = 0.1
    sigma = 0.01

    HHt = H * H'
    XHt = X * H'

    inner_iter = 0
    for iter = 1:max_iter
        inner_iter = iter
        G = W * HHt .- XHt
        W_old = copy(W)

        W_cand = max.(W_old .- alpha .* G, 0.0)
        d = W_cand .- W_old
        normd = norm(d)
        
        if normd / max(1.0, norm(W_old)) < tol
            break
        end
        
        suff_decr = (1-sigma)*dot(G, d) + 0.5*dot(d, d*HHt) <= 0

        if suff_decr
            while suff_decr
                W = copy(W_cand)
                alpha /= beta
                W_cand = max.(W_old .- alpha .* G, 0.0)
                if norm(W - W_cand) <= tol 
                    break
                end
                d = W_cand .- W_old
                suff_decr = (1-sigma)*dot(G, d) + 0.5*dot(d, d*HHt) <= 0
                if alpha > 1e10; break; end
            end
            alpha *= beta
        else
            while !suff_decr
                alpha *= beta
                W_cand = max.(W_old .- alpha .* G, 0.0)
                d = W_cand .- W_old
                suff_decr = (1-sigma)*dot(G, d) + 0.5*dot(d, d*HHt) <= 0
                if alpha < 1e-10; break; end
            end
            W .= W_cand
        end
        
        if norm(W - W_old) / max(1.0, norm(W_old)) < tol
            break
        end
    end
    return W, inner_iter, alpha
end

function projected_gradient_lin_H(X, W, H0; alpha_init = 1.0, tol = 1e-4, max_iter = 50)
    H = copy(H0)
    alpha = alpha_init
    beta = 0.1
    sigma = 0.01

    WtW = W' * W
    WtX = W' * X

    inner_iter = 0
    for iter = 1:max_iter
        inner_iter = iter
        G = WtW * H .- WtX
        H_old = copy(H)

        H_cand = max.(H_old .- alpha .* G, 0.0)
        d = H_cand .- H_old
        normd = norm(d)
        
        if normd / max(1.0, norm(H_old)) < tol
            break
        end
        
        suff_decr = (1-sigma)*dot(G, d) + 0.5*dot(d, WtW*d) <= 0

        if suff_decr
            while suff_decr
                H = copy(H_cand)
                alpha /= beta
                H_cand = max.(H_old .- alpha .* G, 0.0)
                if norm(H - H_cand) <= tol 
                    break
                end
                d = H_cand .- H_old
                suff_decr = (1-sigma)*dot(G, d) + 0.5*dot(d, WtW*d) <= 0
                if alpha > 1e10; break; end
            end
            alpha *= beta
        else
            while !suff_decr
                alpha *= beta
                H_cand = max.(H_old .- alpha .* G, 0.0)
                d = H_cand .- H_old
                suff_decr = (1-sigma)*dot(G, d) + 0.5*dot(d, WtW*d) <= 0
                if alpha < 1e-10; break; end
            end
            H .= H_cand
        end

        if norm(H - H_old) / max(1.0, norm(H_old)) < tol
            break
        end
    end
    return H, inner_iter, alpha
end

# =========================================================================
# Algoritmo Principal
# =========================================================================

function nmf_lin_algorithm(X, r, W_init, H_init; max_iter=100, tol=1e-2, log_io=stdout, log_interval=10)
    sub_max_iter = 1000
    sub_tol = 1e-3
    
    m, n = size(X)
    W = copy(W_init)
    H = copy(H_init)
    errors = Float64[]
    t_start = time()
    total_sub_iters = 0
    alpha_W = 1.0
    alpha_H = 1.0
    
    t_now = Dates.format(now(), "HH:MM:SS")
    println(log_io, "") 
    println(log_io, "[$t_now] [LIN_ALGO] Starting Optimization (MaxIter=$max_iter, Tol=$tol)")
    println(log_io, "[$t_now] [LIN_ALGO] ITER |  RECON_ERROR  |   DELTA_W   |   DELTA_H   | TIME(s)")
    println(log_io, "-------------------------------------------------------------------------------------")

    final_iter = 0
    stop_reason = "Max Iterations Reached"

    for iter = 1:max_iter
        final_iter = iter 
        W_old = copy(W)
        H_old = copy(H)

        W, iter_W, alpha_W = projected_gradient_lin_W(X, H, W; alpha_init=alpha_W, tol=sub_tol, max_iter=sub_max_iter)
        total_sub_iters += iter_W

        H, iter_H, alpha_H = projected_gradient_lin_H(X, W, H; alpha_init=alpha_H, tol=sub_tol, max_iter=sub_max_iter)
        total_sub_iters += iter_H

        current_error = norm(X - W * H) / max(1.0, norm(X))
        push!(errors, current_error)
        
        deltaW = norm(W - W_old) / max(1.0, norm(W_old))
        deltaH = norm(H - H_old) / max(1.0, norm(H_old))

        # --- Lógica de Parada ---
        should_stop = false
        if current_error < tol
            stop_reason = "Converged (Error < $tol)"
            should_stop = true
        elseif deltaW < tol && deltaH < tol
            stop_reason = "Converged (Delta < $tol)"
            should_stop = true
        end

        # --- LOGGING ---
        # Imprime se:
        # 1. É a primeira iteração
        # 2. É uma iteração de intervalo (ex: 20, 40...)
        # 3. OU se vai parar agora (should_stop == true) -> GARANTE A ÚLTIMA LINHA
        if iter == 1 || iter % log_interval == 0 || should_stop
             t_now_iter = Dates.format(now(), "HH:MM:SS")
             elapsed = time() - t_start
             @printf(log_io, "[%s] [LIN_ALGO] %04d | %.6e | %.4e | %.4e | %6.2f\n", 
                     t_now_iter, iter, current_error, deltaW, deltaH, elapsed)
             flush(log_io)
        end

        if should_stop
            break
        end
    end
    
    t_now_end = Dates.format(now(), "HH:MM:SS")
    println(log_io, "[$t_now_end] [LIN_ALGO] STOPPED at Iter $final_iter: $stop_reason")
    println(log_io, "-------------------------------------------------------------------------------------")
    
    return W, H, errors, time() - t_start, total_sub_iters
end