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

function nmf_gradient_projected(X, r, W_init, H_init; max_iter=MAX_ITER_FIXED, tol=TOL_FIXED, 
                                sub_tol=TOL_FIXED_SUB, sub_max_iter=MAX_ITER_FIXED, 
                                monotone=true, alpha_rule_W=(args...)->args[5], alpha_rule_H=(args...)->args[5], kwargs...)
    
    W = copy(W_init); H = copy(H_init)
    errors = Float64[]
    t_start = time()
    total_sub = 0

    for iter = 1:max_iter
        W_old = copy(W); H_old = copy(H)

        W, iW = projected_gradient_W(X, H, W; tol=sub_tol, max_iter=sub_max_iter, monotone=monotone, alpha_rule_W=alpha_rule_W)
        total_sub += iW
        
        H, iH = projected_gradient_H(X, W, H; tol=sub_tol, max_iter=sub_max_iter, monotone=monotone, alpha_rule_H=alpha_rule_H)
        total_sub += iH

        push!(errors, norm(X - W * H))

        if norm(W - W_old)/max(1, norm(W_old)) < tol && norm(H - H_old)/max(1, norm(H_old)) < tol
            break
        end
    end
    return W, H, errors, time() - t_start, total_sub
end