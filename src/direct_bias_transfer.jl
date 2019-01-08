using DataFrames
using Statistics
using Plots
using LinearAlgebra
using SparseArrays
using CSV

src_dir = "/h/mebrunet/Code/understanding-bias/src/"
include(src_dir * "Corpora.jl")
include(src_dir * "GloVe.jl")
include(src_dir * "Bias.jl")
include(src_dir * "utils.jl")

# include("Corpora.jl")
# include("GloVe.jl")
# include("Bias.jl")
# include("utils.jl")


# Load NYT
target = get(ARGS, 1, "C1-V15-W8-D200-R0.05-E150-B1")
# embedding_dir = get(ARGS, 2, "embeddings")
embedding_dir = get(ARGS, 2, "/scratch/gobi2/mebrunet/fat/embeddings")
# pert_dir = get(ARGS, 3, "results/perturbations")
pert_dir = get(ARGS, 3, "/scratch/gobi2/mebrunet/fat/perturbations")

finfo = fileinfo(target)
vocab_filename = "vocab-$(match(r"C[0-9]+-V[0-9]+", target).match).txt"
vocab, ivocab = GloVe.load_vocab(joinpath(embedding_dir, vocab_filename))

weat_idx_sets = [Bias.get_weat_idx_set(set, vocab) for set in Bias.WEAT_WORD_SETS]
all_weat_indices = unique([i for set in weat_idx_sets for inds in set for i in inds])


function get_gender_dir(def_vectors)
    (n, D) = size(def_vectors)
    pairwise_mean_opp = zeros((n, n))
    for i in 0:(div(n, 2) - 1)
        i_s = 2*i+1
        i_e = 2*i+2
        diag_entry = [1/2 -1/2; -1/2 1/2]
        pairwise_mean_opp[i_s:i_e, i_s:i_e] = diag_entry
    end
    normed_centered = mapslices(normalize, pairwise_mean_opp * def_vectors, dims=2)
    F = svd(normed_centered)
    g = F.Vt[1, :]
    return normalize(g)
end


# Definition of Direct bias in the Man is to Computer Programmer paper
function direct_bias(gender_dir::Array{Float64, 1},
        test_vectors::Array{Float64, 2}, c=1)
    normed = mapslices(normalize, test_vectors, dims=2)
    gender_comps = normed * gender_dir
    return sum(x->abs(x)^c, gender_comps) / length(gender_comps)
end


function get_db(vectors_filepath, vocab, ivocab, weat_idx_set, name)
    println("Direct Bias - $vectors_filepath")
    (W, ) = GloVe.load_bin_vectors(vectors_filepath, length(vocab))
    indices = []
    for (idx1, idx2) in zip(weat_idx_set.A, weat_idx_set.B)
        push!(indices, idx1)
        push!(indices, idx2)
    end
    g = get_gender_dir(W[indices, :])

    words = Array{String,1}()
    bias = Array{Float64,1}()
    for idx in [weat_idx_set.S; weat_idx_set.T]
        db = g' * normalize(W[idx, :])
        println(ivocab[idx], ": ", db)
        push!(words, ivocab[idx])
        push!(bias, db)
    end
    return DataFrame(Dict("word"=>words, name=>bias))
end


words = [Bias.WEAT_WORD_SETS[1].S...; Bias.WEAT_WORD_SETS[1].T...]
counts = [round(Int, vocab[word].count / 1000) for word in words]
df = DataFrame(count=counts, word=words)

begin
    local a = 0
    local b = 0
    for filename in readdir(joinpath(pert_dir, target))
        consider = false
        name = ""
        if (startswith(filename, "vectors-aggravate_10000"))
            a += 1
            name = "aggravate_$a"
            consider = true
        elseif (startswith(filename, "vectors-correct_10000"))
            b += 1
            name = "correct_$b"
            consider = true
        end
        if (consider)
            tmp = get_db(joinpath(pert_dir, target, filename), vocab, ivocab, weat_idx_sets[1], name)
            global df = join(df, tmp, on=:word)
        end
    end
end


for seed in 1:10
    filepath = joinpath(embedding_dir, "vectors-$(target[1:end-3])-S$seed.bin")
    name = "baseline_$seed"
    tmp = get_db(filepath, vocab, ivocab, weat_idx_sets[1], name)
    global df = join(df, tmp, on=:word)
end


save_dir = joinpath(src_dir, "results", "direct_bias")
mkpath(save_dir)
CSV.write(joinpath(save_dir, "projections.csv"), df)
