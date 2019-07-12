using SparseArrays
using LinearAlgebra

include("utils.jl")
include("GloVe.jl")
include("Bias.jl")


target = get(ARGS, 1, "C0-V15-W8-D75-R0.05-E300")
pert_dir = get(ARGS, 2, "results/perturbations")
embedding_dir = get(ARGS, 3, "embeddings")
target_biases = get(ARGS, 4, "12")
test_words_dir = get(ARGS, 5, "etc")

WEAT_TEST_FILES = (science_arts=(S="$test_words_dir/science.txt",
                                 T="$test_words_dir/arts.txt"),
                   instruments_weapons=(A="$test_words_dir/pleasant.txt",
                                        B="$test_words_dir/unpleasant.txt"))

WEAT_TEST_SETS = [Bias.make_weat_test_set(Bias.WEAT_WORD_SETS[k], WEAT_TEST_FILES[k],
                  verbose=true) for k in keys(Bias.WEAT_WORD_SETS)]


function main()
    vocab_path = abspath(joinpath(embedding_dir, "vocab-$(split(target, "-W")[1]).txt"))
    println("Vocab: $vocab_path")
    vocab, ivocab = GloVe.load_vocab(vocab_path)
    V = length(vocab)

    weat_idx_sets = [Bias.get_weat_idx_set(set, vocab) for set in WEAT_TEST_SETS]
    all_weat_indices = unique([i for set in weat_idx_sets for inds in set for i in inds])

    for j in target_biases
        i = parse(Int, j)
        target_dir = joinpath(pert_dir, target * "-B$i")
        open(joinpath(target_dir, "test_word_change.csv"), "w") do out_io
            headers = ["filename", "pert_type", "pert_size", "pert_run", "seed", "trueB̃"]
            println(out_io, join(headers,","))
            # Baselines
            for vector_filename in readdir(embedding_dir)
                if startswith(vector_filename, "vectors-$target")
                    println("Baseline Vecs: $vector_filename")
                    finfo = fileinfo(vector_filename)
                    (W, b_w, U, b_u) = GloVe.load_bin_vectors(joinpath(embedding_dir, vector_filename), V)
                    trueB̃ = Bias.effect_size(W, weat_idx_sets[i])
                    data = [vector_filename, "baseline", 0, 1, finfo.seed, trueB̃]
                    println(out_io, join(data, ","))
                end
            end
            # Perturbations
            for vector_filename in readdir(target_dir)
                if startswith(vector_filename, "vectors-")
                    println("Pert Vecs: $vector_filename")
                    fields = split(vector_filename[9:end-4], "_")
                    (W, b_w, U, b_u) = GloVe.load_bin_vectors(joinpath(target_dir, vector_filename), V)
                    trueB̃ = Bias.effect_size(W, weat_idx_sets[i])
                    data = [vector_filename, fields..., trueB̃]
                    println(out_io, join(data, ","))
                end
            end
        end
    end
end

main()
