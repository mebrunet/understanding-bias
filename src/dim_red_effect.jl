using LinearAlgebra
using Plots
using SparseArrays
import Arpack

using Distributed
using SharedArrays
addprocs(4)

include("utils.jl")
include("GloVe.jl")
include("PPMI.jl")
include_everywhere("Bias.jl")

# Globals
RESULTS_DIR = "results/dim_red_effect"

# Define cosine similiarty function
cossim(x, y) = (x' * y) / (norm(x) * norm(y))

# Synthetic data ------------------------
n = 30_000  # Number of datapoints
d = 10_000  # dimension of data

var = sort!(randn(d) .^ 2, rev=true)
plot(var)

# Construct Data - knowing what dimensions contain most variance
D = Diagonal(var) * randn(d, n)

# Select random pairs of vector
num_pairs = 5000
idx_pairs = [rand(1:n, 2) for i in 1:num_pairs]

sims = zeros(num_pairs)
q = 20
sims_q = zeros(num_pairs)
for (k, (i, j)) in enumerate(idx_pairs)
    sims[k] = cossim(D[:,i], D[:, j])
    sims_q[k] = cossim(D[1:q,i], D[1:q, j])
end

histogram(sims, nbins=100, title="Original Space", xlabel="cosine similarity",
          ylabel="count", legend=false)
savefig("$RESULTS_DIR/synth_og.png")
histogram(sims_q, nbins=100, title="Dim Reduced Space",
          xlabel="cosine similarity", ylabel="count", legend=false)
savefig("$RESULTS_DIR/synth_dr.png")
histogram2d(sims, sims_q, nbins=75, xlabel="cossim original",
            ylabel="cossim reduced", title="Pairwise Change")
savefig("$RESULTS_DIR/synth_compare.png")


# GloVe and PPMI ---------------------------
vocab, ivocab = GloVe.load_vocab("embeddings/vocab-C0-V15.txt")
V = length(ivocab)
X = GloVe.load_cooc("embeddings/cooc-C0-V15-W8.bin", V)
W, U, b_w, b_u = GloVe.load_bin_vectors("embeddings/vectors-C0-V15-W8-D75-R0.05-E300-S1.bin", V)

# Form PPMI Matrix
T = PPMI.sum_counts(vocab)
window = 8
r = PPMI.cooc_ratio(window)
M = PPMI.make_ppmi_matrix(X, vocab, ivocab, T, r)

num_eigs=200
(eig_vals, eig_vecs, nconv, niter, nmult, resid) = Arpack.eigs(M, nev=num_eigs)
@assert sum(imag.(eig_vecs)) == 0.0  # No imaginary part
@assert sum(imag.(eig_vals)) == 0.0

A = real.(eig_vecs)
@assert isapprox(sum(mapslices(norm, A, dims=1)), num_eigs)  # Check normalized

# Speed up sparse matrix multiplication
function sparse_mult(A, S, chunk=100; verbose=false)
    q, d = size(A)
    @assert size(S) == (d, d)
    B = zeros(q, d)
    for idx in Iterators.partition(1:d, chunk)
        B[:, idx] = A * Array(S[:, idx])
        verbose && println("$(idx[end]) of $d")
    end
    return B
end


# Dim reduce PPMI
wq=100
Mq = sparse_mult(A[:, 1:wq]', M, 500; verbose=true)


# ----- Cosine Similarity
wnum_pairs = 50_000
widx_pairs = [rand(1:V, 2) for i in 1:wnum_pairs]
for pair in widx_pairs
    if pair[1] == pair[2]
        pair[1] = rand(1:V)
    end
end

wsims = zeros(wnum_pairs)
wsims_q = zeros(wnum_pairs)
wsims_w = zeros(wnum_pairs)

for (k, (i, j)) in enumerate(widx_pairs)
    wsims[k] = cossim(M[:, i], M[:, j])
    wsims_q[k] = cossim(Mq[:, i], Mq[:, j])
    wsims_w[k] = cossim(W[i, :], W[j, :])
end


histogram(wsims, nbins=100, title="Original Space", xlabel="cosine similarity",
          ylabel="count", legend=false)
savefig("$RESULTS_DIR/cossim_PPMI_og.png")
histogram(wsims_q, nbins=100, title="Dim Reduced Space",
          xlabel="cosine similarity", ylabel="count", legend=false)
savefig("$RESULTS_DIR/cossim_PPMI_dr.png")
histogram(wsims_w, nbins=100, title="GloVe",
          xlabel="cosine similarity", ylabel="count", legend=false)
savefig("$RESULTS_DIR/cossim_PPMI_glove.png")
histogram2d(wsims, wsims_q, nbins=200, xlabel="cossim original",
          ylabel="cossim reduced", title="Pairwise Change")
savefig("$RESULTS_DIR/cossim_dr_compare.png")
histogram2d(wsims, wsims_w, nbins=200, xlabel="cossim original",
            ylabel="cossim GloVe", title="Pairwise Change")
savefig("$RESULTS_DIR/cossim_glove_compare.png")

# WEAT ---------------------------

function rand_weat_idx_set(len=8, lim=10_000)
    return (S=rand(1:V, len), T=rand(1:V, len),
            A=rand(1:V, len), B=rand(1:V, len))
end


function make_rand_weat_idx_sets(N; lim=10_000, set_lenghts=8)
    idx_sets = []
    for i in 1:N
        idx_set = rand_weat_idx_set(set_lenghts, lim)
        push!(idx_sets, idx_set)
    end
    return idx_sets
end


num_weat_sets = 20_000
weat_sets = make_rand_weat_idx_sets(num_weat_sets, lim=V)

weat = SharedArray(zeros(num_weat_sets))
weat_q = SharedArray(zeros(num_weat_sets))
weat_w = SharedArray(zeros(num_weat_sets))
@sync @distributed for k in 1:num_weat_sets
    (k % 100) == 0 && println(k)
    weat[k] = Bias.effect_size(M, weat_sets[k])
    weat_q[k] = Bias.effect_size(Mq', weat_sets[k])
    weat_w[k] = Bias.effect_size(W, weat_sets[k])
end


histogram(weat, nbins=100, title="Original Space", xlabel="weat",
          ylabel="count", legend=false)
savefig("$RESULTS_DIR/weat_PPMI_og.png")
histogram(weat_q, nbins=100, title="Dim Reduced Space",
          xlabel="weat", ylabel="count", legend=false)
savefig("$RESULTS_DIR/weat_PPMI_dr.png")
histogram(weat_w, nbins=100, title="GloVe",
          xlabel="weat", ylabel="count", legend=false)
savefig("$RESULTS_DIR/weat_glove.png")

histogram2d(weat, weat_q, nbins=100, xlabel="weat original",
            ylabel="weat reduced", title="Pairwise Change")
savefig("$RESULTS_DIR/weat_dr_compare.png")
histogram2d(weat, weat_w, nbins=100, xlabel="weat original",
            ylabel="weat GloVe", title="Pairwise Change")
savefig("$RESULTS_DIR/weat_glove_compare.png")


# RIPA ----------------------------

# Select random triplets of vector
num_ripa_pairs = 50_000
ripa_pairs = [rand(1:V, 3) for i in 1:num_ripa_pairs]

ripa = zeros(num_ripa_pairs)
ripa_q = zeros(num_ripa_pairs)
ripa_w = zeros(num_ripa_pairs)
for (k, (i, j, z)) in enumerate(ripa_pairs)
    ripa[k] = normalize(M[:, i] - M[:, j])' * M[:, z]
    ripa_q[k] = normalize(Mq[:, i] - Mq[:, j])' * Mq[:, z]
    ripa_w[k] = normalize(W[i, :] - W[j, :])' * W[z, :]
end

histogram(ripa, nbins=100, title="Original Space", xlabel="ripa",
          ylabel="count", legend=false)
savefig("$RESULTS_DIR/ripa_PPMI_og.png")
histogram(ripa_q, nbins=100, title="Dim Reduced Space",
          xlabel="ripa", ylabel="count", legend=false)
savefig("$RESULTS_DIR/ripa_PPMI_dr.png")
histogram(ripa_w, nbins=100, title="GloVe Embedding",
          xlabel="ripa", ylabel="count", legend=false)
savefig("$RESULTS_DIR/ripa_glove.png")
histogram2d(ripa, ripa_q, nbins=200, xlabel="ripa original",
          ylabel="ripa reduced", title="Pairwise Change")
savefig("$RESULTS_DIR/ripa_dr_compare.png")
histogram2d(ripa, ripa_w, nbins=200, xlabel="ripa original",
            ylabel="ripa GloVe", title="Pairwise Change")
savefig("$RESULTS_DIR/ripa_glove_compare.png")
