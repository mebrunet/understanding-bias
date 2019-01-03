using DataFrames
using Statistics
using Plots
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using JSON
using CSV

include("Corpora.jl")
include("GloVe.jl")
include("Bias.jl")
include("PPMI.jl")
include("utils.jl")


target = get(ARGS, 1, "C1-V15-W8-D200-R0.05-E150-B1")
embedding_dir = get(ARGS, 2, "embeddings")
pert_dir = get(ARGS, 3, "results/perturbations")
finfo = fileinfo(target)

# Vocab
vocab_filename = "vocab-$(match(r"C[0-9]+-V[0-9]+", target).match).txt"
vocab, ivocab = GloVe.load_vocab(joinpath(embedding_dir, vocab_filename))

# Embeddings
embeddings = []
for i in 1:5
    vectors_filename = "vectors-$(match(r"C[0-9]+-.*-E[0-9]+", target).match)-S$i.bin"
    (temp, ) = GloVe.load_bin_vectors(joinpath(embedding_dir, vectors_filename), length(vocab))
    push!(embeddings, temp)
end


function rand_weat_set(ivocab; len=8, lim=10_000)
    V = lim > 0 ? lim : length(ivocab)
    return (S=ivocab[rand(1:V, len)], T=ivocab[rand(1:V, len)],
            A=ivocab[rand(1:V, len)], B=ivocab[rand(1:V, len)])
end


function mean_effect_size(embeds, idx_set)
    es = []
    for W in embeds
        push!(es, Bias.effect_size(W, idx_set))
    end
    return mean(es)
end


function make_weat_sets(ivocab; len=8, N=2_000, lim=10_000)
    idx_sets = []
    for i in 1:N
        if i % 100 == 0
            @show i
        end
        idx_set = rand_weat_set(ivocab, len=len, lim=lim)
        push!(idx_sets, idx_set)
    end

    return idx_sets
end


# Random WEAT sets
LIM = 10_000
rand_weat_sets = make_weat_sets(ivocab, lim=LIM)

open("results/weat_compare/rand_weat_sets.json", "w") do f
    JSON.print(f, rand_weat_sets)
end

# Compute the glove effect sizes
glove_effect_sizes = []
for word_set in rand_weat_sets
    idx_set = Bias.get_weat_idx_set(word_set, vocab)
    push!(glove_effect_sizes, mean_effect_size(embeddings, idx_set))
end

df = DataFrame(glove=glove_effect_sizes)
CSV.write("results/weat_compare/glove_weat.csv", df)

# Compute PPMI matrix
cooc_filename = "cooc-$(match(r"C[0-9]+-.*-W[0-9]", target).match).bin"
X = GloVe.load_cooc(joinpath(embedding_dir, cooc_filename), length(vocab), LIM)
r = PPMI.cooc_ratio(finfo.window)
T = PPMI.sum_counts(vocab)
D = PPMI.make_ppmi_matrix(X, vocab, ivocab, T, r)

ppmi_effect_sizes = []
for word_set in rand_weat_sets
    idx_set = Bias.get_weat_idx_set(word_set, vocab)
    push!(ppmi_effect_sizes, Bias.effect_size(D, idx_set))
end

df = DataFrame(ppmi=ppmi_effect_sizes)
CSV.write("results/weat_compare/ppmi_weat.csv", df)

# Raw Cooc effect size
cooc_effect_sizes = []
for word_set in rand_weat_sets
    idx_set = Bias.get_weat_idx_set(word_set, vocab)
    push!(cooc_effect_sizes, Bias.effect_size(X, idx_set))
end

df = DataFrame(cooc=cooc_effect_sizes)
CSV.write("results/weat_compare/cooc_weat.csv", df)



cor(ppmi_effect_sizes, glove_effect_sizes)
