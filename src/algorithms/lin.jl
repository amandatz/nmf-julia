using LinearAlgebra
using Statistics
using Printf
using Dates

# ==========================================================
# Funções Auxiliares
# ==========================================================

function projected_gradient_lin_W(X, H, W0; alpha_init = 1.0, tol = 1e-4, max_iter = 500)
    W = copy(W0)
    alpha = alpha_init
    
    beta = 0.5 
    sigma = 0.01

    HHt = H * H'
    XHt = X * H'

    inner_iter = 0
    for iter = 1:max_iter
        inner_iter = iter
        G = W * HHt .- XHt
        W_old = copy(W)

        # --- Início da Busca Linear ---
        W_cand = max.(W_old .- alpha .* G, 0.0)
        d = W_cand .- W_old
        
        suff_decr = (1-sigma)*sum(G .* d) + 0.5*tr(d * HHt * d') <= 0

        if suff_decr
            while suff_decr
                W = copy(W_cand)
                alpha /= beta
                
                W_cand = max.(W_old .- alpha .* G, 0.0)
                d = W_cand .- W_old
                suff_decr = (1-sigma)*sum(G .* d) + 0.5*tr(d * HHt * d') <= 0
                
                if alpha > 1e10; break; end
            end
            alpha *= beta
        else
            while !suff_decr
                alpha *= beta
                W_cand = max.(W_old .- alpha .* G, 0.0)
                d = W_cand .- W_old
                suff_decr = (1-sigma)*sum(G .* d) + 0.5*tr(d * HHt * d') <= 0
                
                if alpha < 1e-10; break; end
            end
            W .= W_cand
        end
        # --- Fim da Busca Linear ---

        if norm(W - W_old) / max(1.0, norm(W_old)) < tol
            break
        end
    end
    return W, inner_iter, alpha
end

function projected_gradient_lin_H(X, W, H0; alpha_init = 1.0, tol = 1e-4, max_iter = 500)
    H = copy(H0)
    alpha = alpha_init
    beta = 0.5
    sigma = 0.01

    WtW = W' * W
    WtX = W' * X

    inner_iter = 0
    for iter = 1:max_iter
        inner_iter = iter
        G = WtW * H .- WtX
        H_old = copy(H)

        # --- Início da Busca Linear ---
        H_cand = max.(H_old .- alpha .* G, 0.0)
        d = H_cand .- H_old
        
        suff_decr = (1-sigma)*sum(G .* d) + 0.5*tr(d' * WtW * d) <= 0

        # AUMENTA O TAMANHO DE PASSO: ADICIONADO PARA OTIMIZAÇÃO, NÃO TEM NO ARTIGO DO LIN
        if suff_decr
            while suff_decr
                H = copy(H_cand)
                alpha /= beta
                
                H_cand = max.(H_old .- alpha .* G, 0.0)
                d = H_cand .- H_old
                suff_decr = (1-sigma)*sum(G .* d) + 0.5*tr(d' * WtW * d) <= 0
                
                if alpha > 1e10; break; end
            end
            alpha *= beta
        else
            while !suff_decr
                alpha *= beta
                H_cand = max.(H_old .- alpha .* G, 0.0)
                d = H_cand .- H_old
                suff_decr = (1-sigma)*sum(G .* d) + 0.5*tr(d' * WtW * d) <= 0
                
                if alpha < 1e-10; break; end
            end
            H .= H_cand
        end
        # --- Fim da Busca Linear ---

        if norm(H - H_old) / max(1.0, norm(H_old)) < tol
            break
        end
    end
    return H, inner_iter, alpha
end

# ==========================================================
# Algoritmo Principal
# ==========================================================

function nmf_lin_algorithm(
    X, r, W_init, H_init;
    max_iter = 100,
    tol = 1e-5,
    sub_max_iter = 50,
    sub_tol = 1e-4,
    log_io::IO = stdout,
    log_interval::Int = 10
)
    m, n = size(X)
    W = copy(W_init)
    H = copy(H_init)

    errors = Float64[]
    t_start = time()
    total_sub_iters = 0

    timestamp = Dates.format(now(), "HH:MM:SS")
    println(log_io, "[$timestamp] [LIN_ALGO] Starting Optimization (MaxIter=$max_iter, Tol=$tol, Beta=0.5)")
    println(log_io, "[$timestamp] [LIN_ALGO] ITER |  RECON_ERROR  |   DELTA_W  |   DELTA_H  |  ALPHA_W |  ALPHA_H | ITER_SUB_W | ITER_SUB_H | TIME(s)")
    println(log_io, "--------------------------------------------------------------------------------------------------------------------")

    alpha_W = 1.0
    alpha_H = 1.0
    
    converged = false

    for iter = 1:max_iter
        W_old = copy(W)
        H_old = copy(H)

        W, iter_W, alpha_W = projected_gradient_lin_W(X, H, W;
            alpha_init = alpha_W, tol = sub_tol, max_iter = sub_max_iter)
        total_sub_iters += iter_W

        H, iter_H, alpha_H = projected_gradient_lin_H(X, W, H;
            alpha_init = alpha_H, tol = sub_tol, max_iter = sub_max_iter)
        total_sub_iters += iter_H

        current_error = norm(X - W * H)
        push!(errors, current_error)

        deltaW = norm(W - W_old) / max(1.0, norm(W_old))
        deltaH = norm(H - H_old) / max(1.0, norm(H_old))

        if iter == 1 || iter % log_interval == 0
            t_now = Dates.format(now(), "HH:MM:SS")
            elapsed_iter = time() - t_start # Tempo decorrido
            @printf(log_io, "[%s] [LIN_ALGO] %04d | %.6e | %.4e | %.4e | %.2e | %.2e |  %03d  |  %03d  | %6.2f\n", 
                    t_now, iter, current_error, deltaW, deltaH, alpha_W, alpha_H, iter_W, iter_H, elapsed_iter)
            flush(log_io) 
        end

        if deltaW < tol && deltaH < tol
            converged = true
            t_now = Dates.format(now(), "HH:MM:SS")
            elapsed_total = time() - t_start
            println(log_io, "[$t_now] [LIN_ALGO] CONVERGED at Iter $iter (Time: $(round(elapsed_total, digits=2))s)")
            break
        end
    end

    if !converged
        t_now = Dates.format(now(), "HH:MM:SS")
        println(log_io, "[$t_now] [LIN_ALGO] STOPPED: Max Iterations Reached")
    end

    elapsed = time() - t_start
    println(log_io, "--------------------------------------------------------------------------------------------------------------------")
    return W, H, errors, elapsed, total_sub_iters
end