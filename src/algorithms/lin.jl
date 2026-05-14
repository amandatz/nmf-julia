# =========================================================================
# Funções de Otimização (com projeção em [0, M])
# =========================================================================

function projected_gradient_lin_W(X, H, W0, W_max; alpha_init = 1.0, tol = 1e-4, max_iter = 50)
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

        # Projeção no intervalo [0, W_max] (entrada a entrada)
        W_cand = max.(min.(W_old .- alpha .* G, W_max), 0.0)
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
                W_cand = max.(min.(W_old .- alpha .* G, W_max), 0.0)
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
                W_cand = max.(min.(W_old .- alpha .* G, W_max), 0.0)
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

function projected_gradient_lin_H(X, W, H0, H_max; alpha_init = 1.0, tol = 1e-4, max_iter = 50)
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

        # Projeção no intervalo [0, H_max]
        H_cand = max.(min.(H_old .- alpha .* G, H_max), 0.0)
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
                H_cand = max.(min.(H_old .- alpha .* G, H_max), 0.0)
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
                H_cand = max.(min.(H_old .- alpha .* G, H_max), 0.0)
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
# Execução interna (com cotas fixas)
# =========================================================================

function _run_nmf_lin_fixed_cotas(X, r, W_init, H_init, W_max, H_max; 
                                  max_iter=100, tol=1e-2, log_io=stdout, log_interval=10)
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

    t_now = Dates.format(now(), "HH:MM:SS")
    println(log_io, "") 
    println(log_io, "[$t_now] [LIN_FIXED] Starting Optimization (MaxIter=$max_iter, Tol=$tol)")
    println(log_io, "[$t_now] [LIN_FIXED] W_max = $W_max, H_max = $H_max")
    println(log_io, "[$t_now] [LIN_FIXED] ITER |  RECON_ERROR  |   DELTA_W   |   DELTA_H   | TIME(s)")
    println(log_io, "-------------------------------------------------------------------------------------")

    final_iter = 0
    stop_reason = "Max Iterations Reached"

    for iter = 1:max_iter
        final_iter = iter 
        W_old = copy(W)
        H_old = copy(H)

        W, iter_W, alpha_W = projected_gradient_lin_W(X, H, W, W_max; alpha_init=alpha_W, tol=sub_tol, max_iter=sub_max_iter)
        total_sub_iters += iter_W

        H, iter_H, alpha_H = projected_gradient_lin_H(X, W, H, H_max; alpha_init=alpha_H, tol=sub_tol, max_iter=sub_max_iter)
        total_sub_iters += iter_H

        current_error = norm(X - W * H) / max(1.0, norm(X))
        push!(errors, current_error)
        
        deltaW = norm(W - W_old) / max(1.0, norm(W_old))
        deltaH = norm(H - H_old) / max(1.0, norm(H_old))

        if current_error < tol
            stop_reason = "Converged (Error < $tol)"
            break
        elseif deltaW < tol && deltaH < tol
            stop_reason = "Converged (Delta < $tol)"
            break
        end

        if iter == 1 || iter % log_interval == 0
             t_now_iter = Dates.format(now(), "HH:MM:SS")
             elapsed = time() - t_start
             @printf(log_io, "[%s] [LIN_FIXED] %04d | %.6e | %.4e | %.4e | %6.2f\n", 
                     t_now_iter, iter, current_error, deltaW, deltaH, elapsed)
             flush(log_io)
        end
    end
    
    # Contagem de entradas que tocaram a fronteira superior
    tol_bound = 1e-12
    hit_W = count(x -> x >= W_max - tol_bound, W)
    hit_H = count(x -> x >= H_max - tol_bound, H)
    
    t_now_end = Dates.format(now(), "HH:MM:SS")
    println(log_io, "[$t_now_end] [LIN_FIXED] STOPPED at Iter $final_iter: $stop_reason")
    println(log_io, "-------------------------------------------------------------------------------------")
    
    return W, H, errors, time() - t_start, total_sub_iters, hit_W, hit_H
end

# =========================================================================
# Algoritmo Principal com adaptação de cotas (aumento e reinício)
# =========================================================================

function nmf_lin_algorithm(X, r, W_init, H_init; 
                           max_iter=100, tol=1e-2, log_io=stdout, log_interval=10,
                           max_restarts=10, increase_factor=2.0)
    # Cálculo das cotas iniciais (baseado no ponto inicial)
    m, n = size(X)
    W = copy(W_init)
    H = copy(H_init)

    W_max = (sqrt(n) * norm(X, Inf)) / 1e6
    H_max = (sqrt(m) * norm(X, Inf)) / 1e6
    
    if !isfinite(W_max) || W_max < norm(X)/r
        W_max = norm(X) / r
    end
    if !isfinite(H_max) || H_max < norm(X)/r
        H_max = norm(X) / r
    end

    # Loop de adaptação
    total_restarts = 0          # contador de reinícios efetivos
    final_hit_W = 0
    final_hit_H = 0
    final_W = nothing
    final_H = nothing
    final_errors = nothing
    final_time = 0.0
    final_iters = 0

    for restart in 0:max_restarts
        if restart > 0
            println(log_io, "")
            println(log_io, ">>> COTA ATIVADA (restart #$restart): aumentando W_max e H_max por fator $increase_factor")
            println(log_io, ">>> Reiniciando algoritmo a partir do ponto atual...")
            W_max *= increase_factor
            H_max *= increase_factor
            total_restarts = restart    # atualiza contador
        end
        
        W, H, errors, t, iters, hit_W, hit_H = _run_nmf_lin_fixed_cotas(
            X, r, W, H, W_max, H_max;
            max_iter=max_iter, tol=tol,
            log_io=log_io, log_interval=log_interval
        )
        
        final_W = W
        final_H = H
        final_errors = errors
        final_time += t
        final_iters += iters
        final_hit_W = hit_W
        final_hit_H = hit_H
        
        if hit_W == 0 && hit_H == 0
            println(log_io, "Nenhuma cota ativada. Convergência final alcançada após $total_restarts reinício(s).")
            break
        elseif restart == max_restarts
            println(log_io, "ATENÇÃO: Número máximo de reinícios ($max_restarts) atingido. Ainda há cotas ativadas (W hit=$hit_W, H hit=$hit_H).")
            break
        end
    end
    
    # Log adicional com o total de reinícios (já incluso nas mensagens, mas pode repetir)
    println(log_io, "Total de reinícios executados: $total_restarts")
    
    return final_W, final_H, final_errors, final_time, final_iters, final_hit_W, final_hit_H, total_restarts
end