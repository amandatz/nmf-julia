function nmf_multiplicative(X, r, W_init, H_init; max_iter=MAX_ITER_FIXED, tol=TOL_FIXED)
    m, n = size(X)
    W = copy(W_init)
    H = copy(H_init)
    errors = Float64[]
    iters = 0

    t_start = time()
    for iter = 1:max_iter
        print("$iter ")
        
        W_old = copy(W); H_old = copy(H)

        H .= H .* ((W' * X) ./ max.(W' * W * H, 1e-8))
        H .= max.(H, 0.0)
        
        W .= W .* ((X * H') ./ max.(W * (H * H'), 1e-8))
        W .= max.(W, 0.0)

        push!(errors, norm(X - W * H))
        iters = iter

        if norm(W - W_old)/max(1, norm(W_old)) < tol && norm(H - H_old)/max(1, norm(H_old)) < tol
            break
        end
    end
    return W, H, errors, time() - t_start, iters
end