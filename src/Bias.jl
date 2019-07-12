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


# Effect size directly in an unembedded space (e.g. PPMI)
function effect_size(X::SparseMatrixCSC, weat_idx_set::NamedTuple; limit_scope=false)
    indices = [i for inds in weat_idx_set for i in inds]
    S = limit_scope ? X[indices, weat_idx_set.S] : X[:, weat_idx_set.S]
    T = limit_scope ? X[indices, weat_idx_set.T] : X[:, weat_idx_set.T]
    A = limit_scope ? X[indices, weat_idx_set.A] : X[:, weat_idx_set.A]
    B = limit_scope ? X[indices, weat_idx_set.B] : X[:, weat_idx_set.B]
    return effect_size(S, T, A, B)
end


# Get weat vectors from indices, possibly adding small updates (deltas) to them
function make_weat_vec_set(W::AbstractArray, weat_idx_set::NamedTuple;
                           deltas::Dict=Dict{Integer,Array}())::NamedTuple
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
   return NamedTuple{(:S, :T, :A, :B)}(Tuple(weat_vec_set))
end


# Helper to compute effect size after changes to the embedding
function effect_size(W::AbstractArray, weat_idx_set::NamedTuple, deltas::Dict)
    weat_vec_set = make_weat_vec_set(W, weat_idx_set, deltas=deltas)
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


function compute_weights(vocab::Dict, ivocab::Array{String}, indices;
                         family=:prop, α::Real=1/3, β::Real=15)
    raw_counts = [vocab[ivocab[idx]].count for idx in indices]
    if family == :poly
        # Polynomial
        return raw_counts.^α / sum(raw_counts.^α)
    elseif family == :log
        # Logarithmic
        return log.(raw_counts / β) / sum(log.(raw_counts / β))
    else
        # Proportional
        return raw_counts / sum(raw_counts)
    end
end


function make_weat_weight_set(vocab::Dict, ivocab::Array{String},
                              weat_idx_set::NamedTuple; family=:prop,
                              α::Real=1/3, β::Real=15)::NamedTuple
    weights = [compute_weights(vocab, ivocab, indices, family=family, α=α, β=β)
               for indices in weat_idx_set]
    return NamedTuple{(:S, :T, :A, :B)}(Tuple(weights))
end


# Helper to compute weighted effect size after changes to the embedding
function weighted_effect_size(W::AbstractArray, vocab::Dict,
                              ivocab::Array{String}, weat_idx_set::NamedTuple;
                              deltas::Dict=Dict(), family=:prop, α::Real=1/3,
                              β::Real=15)
    weat_vec_set = make_weat_vec_set(W, weat_idx_set, deltas=deltas)
    weat_weight_set = make_weat_weight_set(vocab, ivocab, weat_idx_set,
                                           family=family, α=α, β=β)
    return weighted_effect_size(weat_vec_set, weat_weight_set)
end


function weighted_effect_size(weat_vec_set::NamedTuple,
                              weat_weight_set::NamedTuple)
    Ŝ = normalize_rows(weat_vec_set.S)
    T̂ = normalize_rows(weat_vec_set.T)
    Â = normalize_rows(weat_vec_set.A)
    B̂ = normalize_rows(weat_vec_set.B)

    Cs = weat_weight_set.S
    Ct = weat_weight_set.T
    Ca = weat_weight_set.A
    Cb = weat_weight_set.B

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


# Helper for making WEAT test sets
function replace_weat_words(weat_words, filename; verbose=false)
    if typeof(filename) != String
        # Nothing to overwrite with
        verbose && println("Original word set used.")
        return weat_words
    end
    new_words = unique(readlines(filename, keep=false))
    n = length(new_words)
    filter!(x->!(x in weat_words), new_words)
    m = length(new_words)
    verbose && println("Replaced with $m words. (Removed $(n - m) weat words.)")
    return Tuple(new_words)
end


# Check changes in embedding bias against synonym WEAT words
function make_weat_test_set(weat_word_set::NamedTuple, testfiles::NamedTuple;
                            verbose=false)
    return (
    S=replace_weat_words(weat_word_set.S, get(testfiles, :S, nothing),
                         verbose=verbose),
    T=replace_weat_words(weat_word_set.T, get(testfiles, :T, nothing),
                         verbose=verbose),
    A=replace_weat_words(weat_word_set.A, get(testfiles, :A, nothing),
                         verbose=verbose),
    B=replace_weat_words(weat_word_set.B, get(testfiles, :B, nothing),
                         verbose=verbose))
end
