using DataFrames
using CSV
using JSON

include("utils.jl")
include("Corpora.jl")
include("GloVe.jl")
include("Bias.jl")
include("PPMI.jl")


# Load
vocab_path = get(ARGS, 1, "embeddings/vocab-C0-V15.txt")
corpus_path = get(ARGS, 2, "corpora/simplewikiselect.txt")
cooc_path = get(ARGS, 3, "embeddings/cooc-C0-V15-W8.bin")
out_file_path = get(ARGS, 4, "results/ppmi_baseline_wiki.csv")
pert_dir = get(ARGS, 5, "results/ppmi_perturbations")

window = extract(cooc_path, r"-W[0-9]*\.", trim=(2,1), cast=Int)
vocab, ivocab = GloVe.load_vocab(vocab_path)
V = length(vocab)
corpus = Corpora.Corpus(corpus_path)

weat_idx_sets = [Bias.get_weat_idx_set(set, vocab) for set in Bias.WEAT_WORD_SETS]
all_weat_indices = unique([i for set in weat_idx_sets for inds in set for i in inds])

X = GloVe.load_cooc(cooc_path, V, all_weat_indices)

r = PPMI.cooc_ratio(window)
T = PPMI.sum_counts(vocab)
D = PPMI.make_ppmi_matrix(X, vocab, ivocab, T, r)
B = [Bias.effect_size(D, idx_set) for idx_set in weat_idx_sets]

N = corpus.num_documents
results = zeros(2, N)

# Compute - about 4.5 hours
open(out_file_path, "w") do out_file
    println(out_file, join(["doc_num", "ΔB1", "ΔB2"], ", "))
    @time open(corpus.corpus_path) do corpus_file
        for doc_num in 1:N
            if doc_num % 100 == 0
                println(doc_num)
                flush(out_file)
            end
            doc_text = Corpora.get_text(corpus_file, corpus, doc_num)
            δX = GloVe.doc2cooc(doc_text, vocab, window)
            D̃ = PPMI.make_ppmi_matrix(X, δX, vocab, ivocab, T, r)
            B̃ = [Bias.effect_size(D̃, idx_set) for idx_set in weat_idx_sets]
            println(out_file, join([string(x) for x in [doc_num; B̃ - B]], ", "))
            results[:, doc_num] = B̃

        end
    end
end

# Read results and sort
df = CSV.read(out_file_path, delim=", ", types=[Float64, Float64, Float64])
df.doc_num = round.(Int, df.doc_num)
describe(df.ΔB1)


# Make perturbations
function make_perturbation(name, article_ids, biases, out_dir; verbose=true)
    verbose && println("Making $(name)... ")
    articles = Corpora.get_texts(corpus, article_ids)
    δX = GloVe.docs2cooc(articles, vocab, window)
    bias = sum(biases)
    GloVe.save_coocs(joinpath(out_dir, "pert-$name.bin"), δX)
    verbose && println("Done.\n")
end


# Limit to bias 1
target = "C0-V15-W8-D75-R0.05-E300"
out_dir = joinpath(pert_dir,"$target-B1")
mkpath(out_dir)
set_sizes = [10, 30, 100, 300, 1000]

begin
    sorted_df = sort(df[[:ΔB1, :doc_num]])
    pert_map = Dict("baseline" => [])

    for N in set_sizes
        # Correcting Articles
        name = "correct_$(N)_1"
        article_ids, biases = sorted_df[:doc_num][1:N], sorted_df[:ΔB1][1:N]
        make_perturbation(name, article_ids, biases, out_dir)
        pert_map[name] = article_ids

        # Aggravating Articles
        name = "aggravate_$(N)_1"
        article_ids = sorted_df[:doc_num][end-(N-1):end]
        biases = sorted_df[:ΔB1][end-(N-1):end]
        make_perturbation(name, article_ids, biases, out_dir)
        pert_map[name] = article_ids
    end

    open(joinpath(out_dir, "pert_map.json"), "w") do f
        JSON.print(f, pert_map)
    end
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

make_config_file(target, out_dir)
