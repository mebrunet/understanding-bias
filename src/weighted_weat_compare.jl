using DataFrames
using Statistics
using Plots
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using Random

include("Corpora.jl")
include("GloVe.jl")
include("Bias.jl")
include("utils.jl")

gr()

# Load NYT
target = get(ARGS, 1, "C1-V15-W8-D200-R0.05-E150")
embedding_dir = get(ARGS, 2, "embeddings")

finfo = fileinfo(target)
vocab_filename = "vocab-$(match(r"C[0-9]+-V[0-9]+", target).match).txt"
cooc_filename = "cooc-$(match(r"C[0-9]+-.*-W[0-9]+", target).match).bin"
vectors_filename = "vectors-$(match(r"C[0-9]+-.*-E[0-9]+", target).match)-S1.bin"

vocab, ivocab = GloVe.load_vocab(joinpath(embedding_dir, vocab_filename))
V = length(ivocab)

weat_idx_sets = [Bias.get_weat_idx_set(set, vocab) for set in Bias.WEAT_WORD_SETS]
all_weat_indices = unique([i for set in weat_idx_sets for inds in set for i in inds])

# X = GloVe.load_cooc(joinpath(embedding_dir, cooc_filename), length(vocab))
X = GloVe.load_cooc(joinpath(embedding_dir, "cooc-C1-V15-W8.weat.bin"), length(vocab))
(W, b_w, U, b_u) = GloVe.load_bin_vectors(joinpath(embedding_dir, vectors_filename), length(vocab))


# Experiment with weighted WEAT biases
WEIGHT_FAMILIES = (:poly, :log, :prop)

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


function make_permutation_sets(word_indices::NamedTuple, N=10_000)
    ST = vcat(word_indices.S, word_indices.T)
    boundary = length(word_indices.S)
    perm_sets = []
    for i in 1:N
        perm = randperm(length(ST))
        S = ST[perm[1:boundary]]
        T = ST[perm[boundary+1:end]]
        push!(perm_sets, (S=S, T=T, A=word_indices.A, B=word_indices.B))
    end
    return perm_sets
end


function plot_trials(trials; title=nothing, savefile=nothing)
    N = length(trials)
    plt = histogram(trials, bins=100, normalize=false, legend=nothing)
    (title != nothing) && title!(title)
    xlabel!("Effect Size")
    ylabel!("Number of Trials (N=$(N))")
    typeof(savefile) == String && savefig(plt, savefile)
    display(plt)
end


function make_family_name(family::Symbol, α=nothing, β=nothing)
    local weat_name
    if family == :prop
        weat_name = "Weighted-$(string(family))"
    elseif family == :log
        weat_name = "Weighted-$(string(family)) (beta=$(round(β, digits=2)))"
    elseif family == :poly
        weat_name = "Weighted-$(string(family)) (alpha=$(round(α, digits=2)))"
    end
    return weat_name
end


function make_save_name(family, α=nothing, β=nothing, sets_name=nothing)
    local save_name
    if family == :prop
        save_name = "trials-$(sets_name)-$(string(family))"
    elseif family == :log
        save_name = "trials-$(sets_name)-$(string(family))_$(round(β, digits=2))"
    elseif family == :poly
        save_name = "trials-$(sets_name)-$(string(family))_$(round(α, digits=2))"
    else
        save_name = "trials-$sets_name-standard"
    end
    return save_name
end


function run_plot_trials(W, vocab, ivocab, idx_sets; sets_name=nothing,
                         family=nothing, α=1/3, β=15, savefolder=nothing)
    local weat_name, trials
    if family == nothing
        weat_name = "Standard"
        trials = [Bias.effect_size(W, idx_set) for idx_set in idx_sets]
    else
        weat_name = make_family_name(family, α, β)
        trials = [Bias.weighted_effect_size(W, vocab, ivocab, idx_set,
                  family=family, α=α, β=β) for idx_set in idx_sets]
    end
    title = "$(weat_name) WEAT"
    if sets_name != nothing
        title *= " on $(sets_name)"
    end
    savefile = nothing
    if typeof(savefolder) == String
        savefile = joinpath(savefolder, make_save_name(family, α, β, sets_name))
    end
    plot_trials(trials, title=title, savefile=savefile)
end


function run_plot_all_trials(W, vocab, ivocab, sets_dict; family=nothing,
                             α=1/3, β=15, savefolder=nothing)
    for (name, idx_sets) in pairs(sets_dict)
        println("Plotting $(name)...")
        run_plot_trials(W, vocab, ivocab, idx_sets, sets_name=name,
                        family=family, α=α, β=β, savefolder=savefolder)
    end
end


function plot_counts(vocab::Dict, ivocab::Array{String}, idx_set;
                     savefile=nothing)
    indices = vcat(idx_set...)
    raw_counts = [vocab[ivocab[idx]].count for idx in indices]
    plt = bar(1:length(indices), raw_counts, yscale=:log10, legend=nothing)
    xticks!(collect(1:length(indices)), ivocab[indices], xrotation=45)
    title!("Word Counts")
    typeof(savefile) == String && savefig(plt, savefile)
    display(plt)
end


function total_counts(word_list, vocab)
    total = 0
    for word in word_list
        total += vocab[word].count
    end
    return total
end


function plot_weights(vocab::Dict, ivocab::Array{String}, idx_set; family=:prop,
                     α=1/3, β=15, savefile=nothing)
    indices = vcat(idx_set...)
    centers = collect(1:length(indices))
    raw_counts = [vocab[ivocab[idx]].count for idx in indices]
    weight_set = Bias.make_weat_weight_set(vocab, ivocab, idx_set, α=α, β=β,
                                            family=family)
    weights = vcat(weight_set...)
    plt = bar(centers, weights, label="weights", color=:blue, legend=false,
              yguide="weights")
    bar!(twinx(), raw_counts, label="counts", yscale=:log10, color=:red,
         opacity=0.3, legend=false)
    xticks!(centers, ivocab[indices], xrotation=45)
    title!(make_family_name(family, α, β))
    typeof(savefile) == String && savefig(plt, savefile)
    display(plt)
end


# (Not)-fun-facts
male_pronoun_count = total_counts(("he", "him", "his"), vocab)
female_pronoun_count = total_counts(("she", "her", "hers"), vocab)
male_pronoun_count / female_pronoun_count


# Make test idx sets
rand_sets = Dict(
    "rand_8" => make_rand_weat_idx_sets(10_000; lim=100_000, set_lenghts=8),
    "rand_25" => make_rand_weat_idx_sets(10_000; lim=100_000, set_lenghts=25),
    "perm_1" => make_permutation_sets(weat_idx_sets[1], 10_000),
    "perm_2" => make_permutation_sets(weat_idx_sets[2], 10_000))


save_dir = "results/weighted_weat"
println("Standard WEAT")
plot_counts(vocab, ivocab, weat_idx_sets[1], savefile="$save_dir/weat1_counts")
run_plot_all_trials(W, vocab, ivocab, rand_sets, savefolder=save_dir)

println("Proportional Weighted WEAT")
plot_weights(vocab, ivocab, weat_idx_sets[1], family=:prop,
             savefile="$save_dir/weights-prop")
run_plot_all_trials(W, vocab, ivocab, rand_sets, family=:prop,
                    savefolder=save_dir)

println("Log Weighted WEAT")
for β in [1, 15]
    plot_weights(vocab, ivocab, weat_idx_sets[1], family=:log, β=β,
                 savefile="$save_dir/weights-log_$(round(β, digits=2))")
    println("beta = $β")
    run_plot_all_trials(W, vocab, ivocab, rand_sets, family=:log, β=β,
                        savefolder=save_dir)
end

println("Polynomial Weighted WEAT")
for α in [1/5, 1/3, 1/2]
    plot_weights(vocab, ivocab, weat_idx_sets[1], family=:poly, α=α,
                 savefile="$save_dir/weights-poly_$(round(α, digits=2))")
    println("alpha = $(round(α, digits=2))")
    run_plot_all_trials(W, vocab, ivocab, rand_sets, family=:poly, α=α,
                        savefolder=save_dir)
end
