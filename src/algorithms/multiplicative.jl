using LinearAlgebra
using Statistics
using Printf
using Dates

function nmf_multiplicative(
    X, r, W_init, H_init;
    max_iter = 200,
    tol = 1e-4,
    log_io::IO = stdout,    # Onde escrever o log (padrão: tela)
    log_interval::Int = 10  # Frequência do log
)
    m, n = size(X)
    W = copy(W_init)
    H = copy(H_init)
    errors = Float64[]
    iters = 0

    t_start = time()

    # --- CABEÇALHO DO LOG ---
    timestamp = Dates.format(now(), "HH:MM:SS")
    println(log_io, "[$timestamp] [MULT_ALGO] Starting Optimization (MaxIter=$max_iter, Tol=$tol)")
    println(log_io, "[$timestamp] [MULT_ALGO] ITER |  RECON_ERROR  |   DELTA_W  |   DELTA_H  | TIME(s)")
    println(log_io, "-------------------------------------------------------------------------------------")

    converged = false

    for iter = 1:max_iter
        iters = iter
        
        W_old = copy(W)
        H_old = copy(H)

        # --- Atualização de H ---
        # H = H .* (W'X) ./ (W'WH + eps)
        numerator_H = W' * X
        denominator_H = (W' * W * H) .+ 1e-9
        H .= H .* (numerator_H ./ denominator_H)
        
        # --- Atualização de W ---
        # W = W .* (XH') ./ (W(HH') + eps)
        numerator_W = X * H'
        denominator_W = (W * (H * H')) .+ 1e-9
        W .= W .* (numerator_W ./ denominator_W)

        # --- Métricas de Convergência ---
        current_error = norm(X - W * H)
        push!(errors, current_error)

        deltaW = norm(W - W_old) / max(1.0, norm(W_old))
        deltaH = norm(H - H_old) / max(1.0, norm(H_old))

        # --- Log Periódico ---
        if iter == 1 || iter % log_interval == 0
            t_now = Dates.format(now(), "HH:MM:SS")
            elapsed_iter = time() - t_start # Tempo decorrido até agora
            @printf(log_io, "[%s] [MULT_ALGO] %04d | %.6e | %.4e | %.4e | %6.2f\n", 
                    t_now, iter, current_error, deltaW, deltaH, elapsed_iter)
            flush(log_io)
        end

        # --- Critério de Parada ---
        if deltaW < tol && deltaH < tol
            converged = true
            t_now = Dates.format(now(), "HH:MM:SS")
            elapsed_total = time() - t_start
            println(log_io, "[$t_now] [MULT_ALGO] CONVERGED at Iter $iter (Time: $(round(elapsed_total, digits=2))s)")
            break
        end
    end

    if !converged
        t_now = Dates.format(now(), "HH:MM:SS")
        println(log_io, "[$t_now] [MULT_ALGO] STOPPED: Max Iterations Reached ($max_iter)")
    end

    elapsed = time() - t_start
    println(log_io, "-------------------------------------------------------------------------------------")
    
    return W, H, errors, elapsed, iters
end