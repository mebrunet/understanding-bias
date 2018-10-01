using SparseArrays
using LinearAlgebra

include("utils.jl")
include("GloVe.jl")
include("Bias.jl")


target = get(ARGS, 1, "C0-V15-W8-D75-R0.05-E300")
pert_dir = get(ARGS, 2, "results/perturbations")
embedding_dir = get(ARGS, 3, "embeddings")


function pre_compute(M, X, word_indices)
    num_words = length(word_indices)
    H = Dict{Int64,Array{Float64,2}}()
    G = Dict{Int64,Array{Float64,1}}()
    for i in word_indices
        H[i] = inv(GloVe.∇²Li(M.U, X, i))
        G[i] = GloVe.∇Li(M.W, M.b_w, M.U, M.b_u, X, i)
    end
    return H, G
end


function predict_bias(pert_path, M, X, inv_hessians, gradients, weat_idx_set)

    Y = GloVe.load_cooc(pert_path, M.V)
    # Make the IF approximation
    target_indices = unique([i for inds in weat_idx_set for i in inds])
    deltas = GloVe.compute_IF_deltas(Y, M, X, target_indices, inv_hessians, gradients)
    # Compute the bias change
    return Bias.effect_size(M.W, weat_idx_set, deltas)
end


function main()
    vocab_path = abspath(joinpath(embedding_dir, "vocab-$(split(target, "-W")[1]).txt"))
    println("Vocab: $vocab_path")
    vocab, ivocab = GloVe.load_vocab(vocab_path)

    weat_idx_sets = [Bias.get_weat_idx_set(set, vocab) for set in Bias.WEAT_WORD_SETS]
    all_weat_indices = unique([i for set in weat_idx_sets for inds in set for i in inds])

    cooc_path = abspath(joinpath(embedding_dir, "cooc-$(split(target, "-D")[1]).bin"))
    println("Cooc: $cooc_path")
    X = GloVe.load_cooc(cooc_path, length(vocab), all_weat_indices)

    n = 0
    for model_filename in readdir(embedding_dir)
        if occursin(Regex("$target-S[0-9]+.bin"), model_filename)
            n += 1
            println("Model: $model_filename")
            M = GloVe.load_model(joinpath(embedding_dir, model_filename))
            @assert M.vocab_path == vocab_path
            H, G = pre_compute(M, X, all_weat_indices)
            effect_sizes = [Bias.effect_size(M.W, set) for set in weat_idx_sets]

            for i in 1:length(Bias.WEAT_WORD_SETS)
                target_dir = joinpath(pert_dir, target * "-B$i")
                if (n == 1)
                    open(joinpath(target_dir, "predicted_change.csv"), "w") do out_io
                        headers = ["model_file","pert_file", "pert_type", "pert_size", "pert_run", "seed", "B", "IFB̃"]
                        println(out_io, join(headers,","))
                    end
                end
                open(joinpath(target_dir, "predicted_change.csv"), "a") do out_io
                    for pert_filename in readdir(target_dir)
                        if startswith(pert_filename, "pert-")
                            println("Pert: $pert_filename")
                            fields = split(pert_filename[6:end-4], "_")
                            pert_type = fields[1]
                            pert_size = parse(Int, fields[2])
                            pert_run = parse(Int, fields[3])
                            seed = parse(Int, match(r"-S[0-9]+", model_filename).match[3:end])
                            pert_path = joinpath(target_dir, pert_filename)
                            IFB̃ = predict_bias(pert_path, M, X, H, G, weat_idx_sets[i])
                            data = [model_filename, pert_filename, pert_type, pert_size, pert_run, seed, effect_sizes[i], IFB̃]
                            println(out_io, join(data, ","))
                        end
                    end
                end
            end
        end
    end

end

main()
