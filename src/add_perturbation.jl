using SparseArrays

include("GloVe.jl")


cooc_file = ARGS[1]
pert_file = ARGS[2]
out_file = ARGS[3]
vocab_file = ARGS[4]

vocab, ivocab = GloVe.load_vocab(vocab_file)
V = length(vocab)

δX = GloVe.load_cooc(pert_file, V)
@show nnz(δX)
println("here")

GloVe.perturb_coocs(cooc_file, out_file, -δX)
