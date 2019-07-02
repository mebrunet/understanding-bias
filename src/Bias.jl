module Bias

using LinearAlgebra
using Statistics
using Random
using SparseArrays

include("word_sets.jl")

export WEAT_WORD_SETS


function get_weat_idx_set(word_set::NamedTuple, vocab::Dict)
    words2indices(words) = [vocab[w].index for w in words if w in keys(vocab)]
    return (S=words2indices(word_set.S), T=words2indices(word_set.T),
            A=words2indices(word_set.A), B=words2indices(word_set.B))
end


function normalize_rows(X::AbstractArray)
    return mapslices(normalize, X, dims=2)
end


function normalize_rows(X::SparseMatrixCSC)
    return mapslices(normalize, X, dims=1)'
end


function effect_size(S::AbstractArray, T::AbstractArray, A::AbstractArray,
        B::AbstractArray)
    Ŝ = normalize_rows(S)
    T̂ = normalize_rows(T)
    Â = normalize_rows(A)
    B̂ = normalize_rows(B)

    μSA = mean(Ŝ * Â', dims=2)
    μSB = mean(Ŝ * B̂', dims=2)
    μTA = mean(T̂ * Â', dims=2)
    μTB = mean(T̂ * B̂', dims=2)

    dS = μSA - μSB
    dT = μTA - μTB
    return (mean(dS) - mean(dT)) / std(vcat(dS, dT))
end


# Helper to grab word vecs for you
function effect_size(W, weat_idx_set::NamedTuple)
    S = W[weat_idx_set.S, :]
    T = W[weat_idx_set.T, :]
    A = W[weat_idx_set.A, :]
    B = W[weat_idx_set.B, :]
    return effect_size(S, T, A, B)
end


function effect_size(X::SparseMatrixCSC, weat_idx_set::NamedTuple; limit_scope=false)
    indices = [i for inds in weat_idx_set for i in inds]
    S = limit_scope ? X[indices, weat_idx_set.S] : X[:, weat_idx_set.S]
    T = limit_scope ? X[indices, weat_idx_set.T] : X[:, weat_idx_set.T]
    A = limit_scope ? X[indices, weat_idx_set.A] : X[:, weat_idx_set.A]
    B = limit_scope ? X[indices, weat_idx_set.B] : X[:, weat_idx_set.B]
    return effect_size(S, T, A, B)
end


# Helper to compute effect size after changes to the embedding
function effect_size(W, weat_idx_set::NamedTuple, deltas::Dict)
    weat_vec_set = []
    delta_indices = keys(deltas) # the indices that have changes
    for indices in weat_idx_set
        vecs = W[indices, :]
        for (idx, pos) = zip(delta_indices, indexin(delta_indices, indices))
            # idx: word index of changed vectors
            # pos: relative position of that index in the "vecs" matrix
            if pos != nothing
                vecs[pos, :] += deltas[idx]
            end
        end
        push!(weat_vec_set, vecs)
    end
    return effect_size(weat_vec_set...)
end


# Compute p-value of the bias
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


function compute_weights(vocab, ivocab, indices; family=:poly, α=1/3, β=15)
    raw_counts = [vocab[ivocab[idx]].count for idx in indices]
    if family == :prop
        return raw_counts / sum(raw_counts)
    elseif family == :log
        return log.(raw_counts / β) / sum(log.(raw_counts / β))
    else
        return raw_counts.^α / sum(raw_counts.^α)
    end
end

function weighted_effect_size(W, vocab, ivocab, weat_idx_set::NamedTuple;
                              family=:poly, α=1/3, β=15)
    Ŝ = normalize_rows(W[weat_idx_set.S, :])
    T̂ = normalize_rows(W[weat_idx_set.T, :])
    Â = normalize_rows(W[weat_idx_set.A, :])
    B̂ = normalize_rows(W[weat_idx_set.B, :])

    Cs = compute_weights(vocab, ivocab, weat_idx_set.S; family=family, α=α, β=β)
    Ct = compute_weights(vocab, ivocab, weat_idx_set.T; family=family, α=α, β=β)
    Ca = compute_weights(vocab, ivocab, weat_idx_set.A; family=family, α=α, β=β)
    Cb = compute_weights(vocab, ivocab, weat_idx_set.B; family=family, α=α, β=β)

    SA = Ŝ * Â'
    SB = Ŝ * B̂'
    TA = T̂ * Â'
    TB = T̂ * B̂'

    sa = SA * Ca
    sb = SB * Cb
    ta = TA * Ca
    tb = TB * Cb

    Ws = Cs
    Wt = Ct
    Wst = vcat(Ws, Wt)
    L = length(Wst)

    μS = Ws' * (sa - sb)
    μT = Wt' * (ta - tb)
    μ = (μS + μT) / 2.0

    σ = √((Ws' * ((sa - sb) .- μ).^2 + Wt' * ((ta - tb) .- μ).^2) / ((L-1)/L * sum(Wst)))

    return (μS - μT) / σ
end

end
