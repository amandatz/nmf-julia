# --- Barzilai–Borwein (Spectral) ---
function make_rule_spectral_W()
    prev_W = nothing
    prev_G = nothing

    return (W, H, G, iter, alpha_prev) -> begin
        alpha = alpha_prev
        if prev_W !== nothing && prev_G !== nothing
            s = W .- prev_W
            y = G .- prev_G
            den = sum(s .* y)
            if den > 0 && isfinite(den)
                alpha = clamp(sum(s .* s) / den, 1e-12, 1e12)
            end
        end
        prev_W, prev_G = copy(W), copy(G)
        return alpha
    end
end

function make_rule_spectral_H()
    prev_H = nothing
    prev_G = nothing

    return (W, H, G, iter, alpha_prev) -> begin
        alpha = alpha_prev
        if prev_H !== nothing && prev_G !== nothing
            s = H .- prev_H
            y = G .- prev_G
            den = sum(s .* y)
            if den > 0 && isfinite(den)
                alpha = clamp(sum(s .* s) / den, 1e-12, 1e12)
            end
        end
        prev_H, prev_G = copy(H), copy(G)
        return alpha
    end
end

# --- Armijo / Line Search Genérico ---
function line_search_segment!(A_old, A_new, G, fixed_mat, f; 
                              beta = 0.5, sigma = 0.1, theta_min = 1e-12, 
                              monotone = true, f_hist = nothing, M_hist = 5)
    dW = A_new .- A_old
    inner = sum(G .* dW)
    f_old = f(A_old, fixed_mat)

    if monotone
        f_ref = f_old
    else
        if f_hist === nothing || isempty(f_hist)
            f_hist = [f_old]
        end
        f_ref = maximum(f_hist)
    end

    theta = 1.0
    A_new_temp = copy(A_new) # Evitar modificar in-place sem querer antes da hora
    f_new = f(A_new_temp, fixed_mat)

    while f_new > f_ref - sigma * theta * inner
        theta *= beta
        if theta < theta_min; break; end
        A_new_temp .= A_old .+ theta .* dW
        f_new = f(A_new_temp, fixed_mat)
    end
    
    # Atualiza o A_new original com o passo aceito
    A_new .= A_new_temp

    if !monotone
        push!(f_hist, f_new)
        if length(f_hist) > M_hist; popfirst!(f_hist); end
    end

    return A_new, theta, f_hist
end