module Bias

using LinearAlgebra, Statistics, Random

include("word_sets.jl")

export WORD_SETS


function get_word_indices(word_set, vocab::Dict)
    words2indices(words) = [vocab[w].index for w in words if w in keys(vocab)]
    return (S=words2indices(word_set.S), T=words2indices(word_set.T),
            A=words2indices(word_set.A), B=words2indices(word_set.B))
end


function normalize_rows(X::Array{Float64,2})
    return mapslices(normalize, X, dims=2)
end


function effect_size(W, word_indices)
    Ŝ = normalize_rows(W[word_indices.S, :])
    T̂ = normalize_rows(W[word_indices.T, :])
    Â = normalize_rows(W[word_indices.A, :])
    B̂ = normalize_rows(W[word_indices.B, :])

    μSA = mean(Ŝ * Â', dims=2)
    μSB = mean(Ŝ * B̂', dims=2)
    μTA = mean(T̂ * Â', dims=2)
    μTB = mean(T̂ * B̂', dims=2)

    dS = μSA - μSB
    dT = μTA - μTB
    return (mean(dS) - mean(dT)) / std(vcat(dS, dT))
end


function p_value(W, word_indices, N=10_000)
    es = effect_size(W, word_indices)
    ST = vcat(word_indices.S, word_indices.T)
    boundary = length(word_indices.S)
    trials = zeros(N)
    for i in 1:N
        perm = randperm(length(ST))
        S = ST[perm[1:boundary]]
        T = ST[perm[boundary+1:end]]
        trials[i] = effect_size(W, (S=S, T=T, A=word_indices.A, B=word_indices.B))
    end
    return mean(trials .> es)
end


end
