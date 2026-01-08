# Funções auxiliares locais
function projected_gradient_lin_W(X, H, W0; alpha_init=1.0, tol=1e-4, max_iter=50)
    W = copy(W0)
    alpha = alpha_init
    beta, sigma = 0.1, 0.01
    HHt = H * H'; XHt = X * H'

    inner_iter = 0
    for iter = 1:max_iter
        inner_iter = iter
        G = W * HHt .- XHt
        W_old = copy(W)
        
        # Backtracking
        while true
            W_cand = max.(W_old .- alpha .* G, 0.0)
            d = W_cand .- W_old
            suff_decr = (1-sigma)*sum(G .* d) + 0.5*tr(d * HHt * d') <= 0
            
            if suff_decr
                W .= W_cand
                break 
            end
            alpha *= beta
            if alpha < 1e-10; break; end
        end
        
        if norm(W - W_old)/max(1.0, norm(W_old)) < tol; break; end
    end
    return W, inner_iter, alpha
end

function projected_gradient_lin_H(X, W, H0; alpha_init=1.0, tol=1e-4, max_iter=50)
    H = copy(H0)
    alpha = alpha_init
    beta, sigma = 0.1, 0.01
    WtW = W' * W; WtX = W' * X

    inner_iter = 0
    for iter = 1:max_iter
        inner_iter = iter
        G = WtW * H .- WtX
        H_old = copy(H)

        while true
            H_cand = max.(H_old .- alpha .* G, 0.0)
            d = H_cand .- H_old
            suff_decr = (1-sigma)*sum(G .* d) + 0.5*tr(d' * WtW * d) <= 0
            
            if suff_decr
                H .= H_cand
                break
            end
            alpha *= beta
            if alpha < 1e-10; break; end
        end

        if norm(H - H_old)/max(1.0, norm(H_old)) < tol; break; end
    end
    return H, inner_iter, alpha
end

# Função Principal do Algoritmo de Lin
function nmf_lin_algorithm(X, r, W_init, H_init; max_iter=100, tol=1e-5, sub_max_iter=50, sub_tol=1e-4)
    W = copy(W_init); H = copy(H_init)
    errors = Float64[]
    t_start = time()
    total_sub = 0
    alpha_W, alpha_H = 1.0, 1.0

    for iter = 1:max_iter
        print("$iter ")

        W_old = copy(W); H_old = copy(H)

        W, iW, alpha_W = projected_gradient_lin_W(X, H, W; alpha_init=alpha_W, tol=sub_tol, max_iter=sub_max_iter)
        total_sub += iW

        H, iH, alpha_H = projected_gradient_lin_H(X, W, H; alpha_init=alpha_H, tol=sub_tol, max_iter=sub_max_iter)
        total_sub += iH

        push!(errors, norm(X - W * H))

        if norm(W - W_old)/max(1, norm(W_old)) < tol && norm(H - H_old)/max(1, norm(H_old)) < tol
            break
        end
    end
    return W, H, errors, time() - t_start, total_sub
end