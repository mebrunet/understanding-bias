# Script to:
# - load diff bias csv files,
# - average to find most biased document sets,
# - create perturbation cooc files for those document sets

using DataFrames
using CSV
using Statistics
using SparseArrays
using LinearAlgebra
using Random

include("Corpora.jl")
include("GloVe.jl")
include("Bias.jl")
include("utils.jl")


# Command ARGS
const target = get(ARGS, 1, "C0-V15-W8-D75-R0.05-E300")
const diff_bias_dir = get(ARGS, 2, "results/diff_bias")
const embedding_dir = get(ARGS, 3, "embeddings")
const corpus_dir = get(ARGS, 4, "corpora")
const pert_dir = get(ARGS, 5, "results/perturbations")


const corpus_prefix = match(r"C[0-9]+-", target).match == "C0-" ? "simplewiki" : "nyt"
const corpus_path = joinpath(corpus_dir, "$(corpus_prefix)select.txt")
const corpus = Corpora.Corpus(corpus_path)

const vocab_prefix = match(r"C[0-9]+-V[0-9]+", target).match
const vocab_path = joinpath(embedding_dir, "vocab-$vocab_prefix.txt")
const (vocab, ivocab) = GloVe.load_vocab(vocab_path)

const window = parse(Int, match(r"W[0-9]+", target).match[2:end])

const set_sizes = corpus_prefix == "nyt" ? [100, 300, 1000, 3000, 10000] : [10, 30, 100, 300, 1000]


# Load a bunch of diff bias results and average them
function load_df(target, bias_col=:ΔBIF_1)
    df = DataFrame()
    i = 0
    cols = []
    for f in readdir(diff_bias_dir)
        if occursin(Regex(target), f)
            i += 1
            tmp = sort(CSV.read(joinpath(diff_bias_dir, f)), :doc_num)
            if (i == 1)
                df[:doc_num] = tmp[:doc_num]
            end
            col = Symbol("ΔBIF_$(i)")
            df[col] = tmp[bias_col]
            push!(cols, col)
        end
    end
    println("Loaded $i differential bias results")
    df[:ΔBIF_μ] = [mean([row[col] for col in cols]) for row in eachrow(df)]
    df[:ΔBIF_σ] = [std([row[col] for col in cols]) for row in eachrow(df)]
    return df
end


function make_perturbation(name, article_ids, biases, out_dir; verbose=true)
    verbose && println("Making $(name)... ")
    articles = Corpora.get_texts(corpus, article_ids)
    δX = GloVe.docs2cooc(articles, vocab, window)
    bias = sum(biases)
    data = (name, nnz(δX), vecnorm(δX), bias)
    GloVe.save_coocs(joinpath(out_dir, "pert-$name.bin"), δX)
    verbose && println("Done.\n")
end


function make_perturbations(target, bias_col=:ΔBIF_1, dir_name="B1")
    out_dir = joinpath(pert_dir, dir_name)
    mkpath(out_dir)
    df = load_df(target, bias_col)
    sorted = sort(df[[:ΔBIF_μ, :doc_num]])

    for N in set_sizes
        # Correcting Articles
        name = "correct_$(N)_1"
        article_ids, biases = sorted[:doc_num][1:N], sorted[:ΔBIF_μ][1:N]
        make_perturbation(name, article_ids, biases, out_dir)

        # Aggravating Articles
        name = "aggravate_$(N)_1"

        article_ids = sorted[:doc_num][end-(N-1):end]
        biases = sorted[:ΔBIF_μ][end-(N-1):end]
        make_perturbation(name, article_ids, biases, out_dir)

        # Random
        all_article_ids = Array{Int64}(df[:doc_num])
        for run in 1:3
            name = "random_$(N)_$(run)"
            article_ids = shuffle(all_article_ids)[1:N]
            biases = [df[:ΔBIF_μ][df[:doc_num] .== id][1] for id in article_ids]
            make_perturbation(name, article_ids, biases, out_dir)
        end
    end
    return out_dir
end


function make_config_file(target, out_dir)
    params = fileinfo("-" * target * "-")
    open(joinpath(out_dir, "config.txt"), "w") do f
        println(f, "CORPUS_ID=$(params.corpus)")
        println(f, "VOCAB_MIN_COUNT=$(params.min_vocab)")
        println(f, "WINDOW_SIZE=$(params.window)")
        println(f, "VECTOR_SIZE=$(params.dimension)")
        println(f, "ETA=$(params.eta)")
        println(f, "MAX_ITER=$(params.max_iters)")
    end
end


function main()
    for i in 1:length(Bias.WEAT_WORD_SETS)
        out_dir = make_perturbations(target, Symbol("ΔBIF_$i"), "$target-B$i")
        make_config_file(target, out_dir)
    end
end


main()
