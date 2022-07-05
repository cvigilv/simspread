using LinearAlgebra

export NBI,
    denovoNBI

function NBI(F₀::AbstractMatrix)
    M, N = size(F₀)
    R = diagm([sum(F₀[i, :]) for i in 1:M])
    H = diagm([sum(F₀[:, j]) for j in 1:N])
    W = (F₀ * H^-1)' * (R^-1 * F₀)

    return W
end

NBI(SymF₀::Symmetric, Nd::Int) = NBI(SymF₀[1:Nd, Nd+1:end])

function denovoNBI(F₀::AbstractMatrix)
    k(A, x) = count(r -> (r != 0), A[x, :]) # Degree helper function
    n_nodes = size(F₀, 1)                   # Number of nodes
    W = zeros(n_nodes, n_nodes)             # Transfer matrix

    for idx in 1:n_nodes
        W[idx, :] = k(F₀, idx) > 0 ? F₀[idx, :] ./ k(F₀, idx) : zeros(n_nodes, 1)
    end

    return W
end
