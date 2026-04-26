# =========================================================================
# Funções de Otimização
# =========================================================================

function projected_gradient_lin_W(X, H, W0; alpha_init = 1.0, tol = 1e-4, max_iter = 50)
    W = copy(W0)
    alpha = 1.0  # sempre local

    HHt = H * H'
    XHt = X * H'

    inner_iter = 0
    for iter = 1:max_iter
        inner_iter = iter
        G = W * HHt .- XHt

        # critério de parada: gradiente projetado
        projgrad = norm(G[G .< 0 .|| W .> 0])
        if projgrad < tol
            break
        end

        W_old = copy(W)
        Wp = copy(W)
        decr_alpha = true

        Wn = max.(W .- alpha .* G, 0.0)
        d = Wn .- W
        gradd = dot(G, d)
        dQd = dot(d, d * HHt)
        suff_decr = 0.99 * gradd + 0.5 * dQd < 0
        decr_alpha = !suff_decr

        for inner_iter = 1:20
            if decr_alpha
                if suff_decr
                    W = Wn; break
                else
                    alpha *= 0.1
                end
            else
                if !suff_decr || Wp == Wn
                    W = Wp; break
                else
                    alpha /= 0.1; Wp = copy(Wn)
                end
            end
            Wn = max.(W .- alpha .* G, 0.0)
            d = Wn .- W
            gradd = dot(G, d)
            dQd = dot(d, d * HHt)
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0
        end
    end
    return W, inner_iter, alpha
end

function projected_gradient_lin_H(X, W, H0; alpha_init = 1.0, tol = 1e-4, max_iter = 50)
    H = copy(H0)
    alpha = 1.0  # sempre local, não reutilizar

    WtW = W' * W
    WtX = W' * X

    inner_iter = 0
    for iter = 1:max_iter
        inner_iter = iter
        G = WtW * H .- WtX

        # critério de parada: gradiente projetado (Lin 2007)
        projgrad = norm(G[G .< 0 .|| H .> 0])
        if projgrad < tol
            break
        end

        H_old = copy(H)
        Hp = copy(H)
        decr_alpha = true

        Hn = max.(H .- alpha .* G, 0.0)
        d = Hn .- H
        gradd = dot(G, d)
        dQd = dot(d, WtW * d)
        suff_decr = 0.99 * gradd + 0.5 * dQd < 0
        decr_alpha = !suff_decr

        for inner_iter = 1:20
            if decr_alpha
                if suff_decr
                    H = Hn; break
                else
                    alpha *= 0.1
                end
            else
                if !suff_decr || Hp == Hn
                    H = Hp; break
                else
                    alpha /= 0.1; Hp = copy(Hn)
                end
            end
            Hn = max.(H .- alpha .* G, 0.0)
            d = Hn .- H
            gradd = dot(G, d)
            dQd = dot(d, WtW * d)
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0
        end
    end
    return H, inner_iter, alpha
end

# =========================================================================
# Algoritmo Principal
# =========================================================================

function nmf_lin_algorithm(X, r, W_init, H_init; max_iter=100, tol=1e-2, log_io=stdout, log_interval=10)
    sub_max_iter = 50
    sub_tol = 1e-3
    
    m, n = size(X)
    W = copy(W_init)
    H = copy(H_init)
    errors = Float64[]
    t_start = time()
    total_sub_iters = 0
    alpha_W = 1.0
    alpha_H = 1.0

    for a in 1:r
        d = sum(W[:, a])
        d = d > 0 ? d : 1.0
        W[:, a] ./= d
        H[a, :] .*= d
    end
    
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

        W, iter_W, _ = projected_gradient_lin_W(X, H, W; alpha_init=alpha_W, tol=sub_tol, max_iter=sub_max_iter)
        total_sub_iters += iter_W

        # Normaliza colunas de W e reescala linhas de H
        for a in 1:r
            d = sum(W[:, a])
            d = d > 0 ? d : 1.0
            W[:, a] ./= d
            H[a, :] .*= d
        end

        H_old = copy(H)

        H, iter_H, _ = projected_gradient_lin_H(X, W, H; alpha_init=alpha_H, tol=sub_tol, max_iter=sub_max_iter)
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