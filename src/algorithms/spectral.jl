using LinearAlgebra
using Statistics
using Printf
using Dates

# ==========================================================
# Regras de Alpha (Barzilai-Borwein)
# ==========================================================

function make_rule_spectral_W()
    prev_W = nothing
    prev_G = nothing
    return (W, G, alpha_prev) -> begin
        alpha = alpha_prev
        if prev_W !== nothing && prev_G !== nothing
            s = W .- prev_W
            y = G .- prev_G
            den = sum(s .* y)
            if den > 1e-12 && isfinite(den)
                # Passo espectral BB1
                alpha = clamp(sum(s .* s) / den, 1e-10, 1.0)
            end
        end
        prev_W = copy(W)
        prev_G = copy(G)
        return alpha
    end
end

function make_rule_spectral_H()
    prev_H = nothing
    prev_G = nothing
    return (H, G, alpha_prev) -> begin
        alpha = alpha_prev
        if prev_H !== nothing && prev_G !== nothing
            s = H .- prev_H
            y = G .- prev_G
            den = sum(s .* y)
            if den > 1e-12 && isfinite(den)
                # Passo espectral BB1
                alpha = clamp(sum(s .* s) / den, 1e-10, 1.0)
            end
        end
        prev_H = copy(H)
        prev_G = copy(G)
        return alpha
    end
end

# ==========================================================
# Subproblemas Otimizados
# ==========================================================

function projected_gradient_W(XHt, HHt, W0; alpha_init=1.0, lambda=0.0, tol=1e-4, max_iter=50, alpha_rule_W=(args...)->args[3])
    W = copy(W0)
    alpha = alpha_init
    
    for iter = 1:max_iter
        W_old = copy(W)
        # G = W*HHt - XHt + lambda*W
        G = W * HHt .- XHt
        if lambda > 0; G .+= lambda .* W; end

        # Sugestão de Alpha via BB
        alpha = alpha_rule_W(W, G, alpha)

        # Busca de linha Armijo Quadrática (Custo O(m * r^2))
        # Não toca na matriz X, usa apenas HHt
        while true
            W_new = max.(W .- alpha .* G, 0.0)
            d = W_new .- W
            
            gradd = dot(G, d)
            # Cálculo eficiente de d' * HHt * d usando propriedades de traço
            dQd = sum((d * HHt) .* d) 
            
            # Condição de Armijo para função quadrática: f(x+d) <= f(x) + sigma * grad'd
            # Aqui simplificamos usando a curvatura exata dQd
            if 0.5 * dQd + gradd <= 0
                W .= W_new
                break
            end
            
            alpha *= 0.5
            if alpha < 1e-12; break; end
        end

        if norm(W .- W_old) / max(1.0, norm(W_old)) < tol
            return W, iter, alpha
        end
    end
    return W, max_iter, alpha
end

function projected_gradient_H(WtX, WtW, H0; alpha_init=1.0, lambda=0.0, tol=1e-4, max_iter=50, alpha_rule_H=(args...)->args[3])
    H = copy(H0)
    alpha = alpha_init
    
    for iter = 1:max_iter
        H_old = copy(H)
        # G = WtW*H - WtX + lambda*H
        G = WtW * H .- WtX
        if lambda > 0; G .+= lambda .* H; end

        # Sugestão de Alpha via BB
        alpha = alpha_rule_H(H, G, alpha)

        while true
            H_new = max.(H .- alpha .* G, 0.0)
            d = H_new .- H
            
            gradd = dot(G, d)
            # Cálculo eficiente de d' * WtW * d
            dQd = sum((WtW * d) .* d)
            
            if 0.5 * dQd + gradd <= 0
                H .= H_new
                break
            end
            
            alpha *= 0.5
            if alpha < 1e-12; break; end
        end

        if norm(H .- H_old) / max(1.0, norm(H_old)) < tol
            return H, iter, alpha
        end
    end
    return H, max_iter, alpha
end

# ==========================================================
# Função Principal
# ==========================================================

function nmf_gradient_projected(X, r, W_init, H_init; 
                                max_iter=500, 
                                tol=1e-4, 
                                sub_tol=1e-4, 
                                sub_max_iter=20, 
                                alpha_rule_W=(args...)->args[3], 
                                alpha_rule_H=(args...)->args[3],
                                alpha_init=1.0,
                                lambda=0.0,
                                log_io::IO = stdout,
                                log_interval::Int = 10,
                                kwargs...)
    
    W = copy(W_init)
    H = copy(H_init)
    errors = Float64[]
    t_start = time()
    total_sub = 0

    curr_alpha_W = alpha_init
    curr_alpha_H = alpha_init
    
    # Pré-calculo constante para critério de erro relativo
    norm_X = norm(X)
    norm_X2 = norm_X^2

    # Normalização inicial (Cota de Lin)
    W_max = norm_X / r
    for a in 1:r
        excess = maximum(W[:, a]) / W_max
        if excess > 1.0
            W[:, a] ./= excess
            H[a, :] .*= excess
        end
    end

    timestamp = Dates.format(now(), "HH:MM:SS")
    @printf(log_io, "[%s] [PG_ALGO] Optimization Started (Fast Mode)\n", timestamp)
    println(log_io, "--------------------------------------------------------------------------------------------------------------------")
    println(log_io, " ITER |  RECON_ERROR  |   DELTA_W  |   DELTA_H  |  ALPHA_W |  ALPHA_H | SUB_W | SUB_H | TIME(s)")
    println(log_io, "--------------------------------------------------------------------------------------------------------------------")

    for iter = 1:max_iter
        W_old = copy(W)
        
        # Pré-computação para subproblema W (O(n * r^2))
        HHt = H * H'
        XHt = X * H'

        W, iW, curr_alpha_W = projected_gradient_W(XHt, HHt, W; 
            tol=sub_tol, max_iter=sub_max_iter,
            alpha_rule_W=alpha_rule_W, alpha_init=curr_alpha_W, lambda=lambda)
        total_sub += iW

        # Transferência de escala (Invariância WH)
        for a in 1:r
            excess = maximum(W[:, a]) / W_max
            if excess > 1.0
                W[:, a] ./= excess
                H[a, :] .*= excess
            end
        end

        H_old = copy(H)
        
        # Pré-computação para subproblema H (O(m * r^2))
        WtW = W' * W
        WtX = W' * X

        H, iH, curr_alpha_H = projected_gradient_H(WtX, WtW, H; 
            tol=sub_tol, max_iter=sub_max_iter,
            alpha_rule_H=alpha_rule_H, alpha_init=curr_alpha_H, lambda=lambda)
        total_sub += iH

        # Cálculo do Erro Global (Apenas nos intervalos de log para poupar O(mnr))
        deltaW = norm(W .- W_old) / max(1.0, norm(W_old))
        deltaH = norm(H .- H_old) / max(1.0, norm(H_old))
        
        # Parada antecipada
        if deltaW < tol && deltaH < tol
            # Cálculo final do erro para o log
            current_error = norm(X - W * H) / norm_X
            t_now = Dates.format(now(), "HH:MM:SS")
            @printf(log_io, "[%s] [PG_ALGO] %04d | %.6e | %.4e | %.4e | CONVERGED\n", t_now, iter, current_error, deltaW, deltaH)
            break
        end

        if iter == 1 || iter % log_interval == 0
            current_error = norm(X - W * H) / norm_X
            push!(errors, current_error)
            t_now = Dates.format(now(), "HH:MM:SS")
            elapsed = time() - t_start
            @printf(log_io, " %04d | %.6e | %.4e | %.4e | %.2e | %.2e |  %02d   |  %02d   | %6.2f\n", 
                    iter, current_error, deltaW, deltaH, curr_alpha_W, curr_alpha_H, iW, iH, elapsed)
            flush(log_io)
        end
    end

    elapsed = time() - t_start
    println(log_io, "--------------------------------------------------------------------------------------------------------------------")
    @printf(log_io, "Finalized in %.2fs. Total sub-iterations: %d\n", elapsed, total_sub)

    return W, H, errors, elapsed, total_sub
end