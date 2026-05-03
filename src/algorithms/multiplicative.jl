function nmf_multiplicative(
    X, r, W_init, H_init;
    max_iter = 200,
    tol = 1e-4,
    log_io::IO = stdout,
    log_interval::Int = 10
)
    m, n = size(X)
    W = copy(W_init)
    H = copy(H_init)
    errors = Float64[]
    iters = 0

    t_start = time()

    timestamp = Dates.format(now(), "HH:MM:SS")
    println(log_io, "[$timestamp] [MULT_ALGO] Starting Optimization (MaxIter=$max_iter, Tol=$tol)")
    println(log_io, "[$timestamp] [MULT_ALGO] ITER |  RECON_ERROR  |   DELTA_W  |   DELTA_H  | TIME(s)")
    println(log_io, "-------------------------------------------------------------------------------------")

    converged = false
    stop_reason = ""

    for iter = 1:max_iter
        iters = iter
        
        W_old = copy(W)
        H_old = copy(H)

        numerator_H = W' * X
        denominator_H = (W' * W * H) .+ 1e-9
        H .= H .* (numerator_H ./ denominator_H)
        
        numerator_W = X * H'
        denominator_W = (W * (H * H')) .+ 1e-9
        W .= W .* (numerator_W ./ denominator_W)

        current_error = norm(X - W * H) / max(1.0, norm(X))
        push!(errors, current_error)

        deltaW = norm(W - W_old) / max(1.0, norm(W_old))
        deltaH = norm(H - H_old) / max(1.0, norm(H_old))

        should_stop = false
        if current_error < tol
            stop_reason = "Converged (Error < $tol)"
            should_stop = true
        elseif deltaW < tol && deltaH < tol
            stop_reason = "Converged (Delta < $tol)"
            should_stop = true
        end

        if iter == 1 || iter % log_interval == 0 || should_stop
            t_now = Dates.format(now(), "HH:MM:SS")
            elapsed_iter = time() - t_start
            @printf(log_io, "[%s] [MULT_ALGO] %04d | %.6e | %.4e | %.4e | %6.2f\n", 
                    t_now, iter, current_error, deltaW, deltaH, elapsed_iter)
            flush(log_io)
        end

        if should_stop
            converged = true
            break
        end
    end

    t_now = Dates.format(now(), "HH:MM:SS")
    if converged
        println(log_io, "[$t_now] [MULT_ALGO] STOPPED: $stop_reason")
    else
        println(log_io, "[$t_now] [MULT_ALGO] STOPPED: Max Iterations Reached ($max_iter)")
    end
    elapsed = time() - t_start
    println(log_io, "-------------------------------------------------------------------------------------")
    
    return W, H, errors, elapsed, iters
end