function generate_matrix(m, n; type=:uniform)
    Q1 = qr(randn(m, m)).Q
    Q2 = qr(randn(n, n)).Q
    d = zeros(min(m, n))
    kappa = 1e8

    if type == :uniform
        d .= rand(length(d))
    elseif type == :decaying
        d .= LinRange(1.0, 0.1, length(d))
    elseif type == :equal
        d .= ones(length(d))
    elseif type == :ill_conditioned
        d .= [1 / kappa^((i - 1) / (length(d) - 1)) for i in 1:length(d)]
    else
        error("Tipo inválido")
    end

    X = abs.(Q1[:, 1:length(d)] * Diagonal(d) * Q2'[1:length(d), :])
    return X
end

function generate_X_WH(m, n, r; type=:uniform)
    W = generate_matrix(m, r; type=type)
    H = generate_matrix(r, n; type=type)
    X = W * H
    return X, W, H
end