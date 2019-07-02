using Test
using SparseArrays

@testset "PPMI" begin

    !isdefined(Main, :GloVe) && include("../src/GloVe.jl")
    include("../src/PPMI.jl")
    test_cooc_path = "test_cooc.bin"
    full_cooc_path = "../embeddings/cooc-C0-V20-W8.bin"
    embedding_path = "../embeddings/vectors-C0-V20-W8-D25-R0.05-E15-S1.bin"
    M = GloVe.load_model(embedding_path)
    X = GloVe.load_cooc(full_cooc_path, M.V)
    δX = GloVe.load_cooc(test_cooc_path, M.V)

    @testset "Cooc to PPMI" begin
        global T = PPMI.sum_counts(M.vocab)
        global r = PPMI.cooc_ratio(M.d)
        global D = PPMI.make_ppmi_matrix(X, M.vocab, M.ivocab, T, r)
        @test nnz(X) == nnz(D)
        @test size(X) == size(D)
    end

    @testset "Perturb PPMI" begin
        X̃ = dropzeros(round.(X - δX, digits=5))
        global D̃ = PPMI.make_ppmi_matrix(X, δX, M.vocab, M.ivocab, T, r)
        true_D̃ = PPMI.make_ppmi_matrix(X̃, M.vocab, M.ivocab, T, r)
        @test isapprox(D̃, true_D̃, rtol=0.001, atol=0.01)
        @test !isapprox(D̃, D, rtol=0.001, atol=0.01)
    end

    @testset "Partial loading of coocs" begin
        indices = rand(1:100, 32)
        Y = GloVe.load_cooc(full_cooc_path, M.V, indices)  # load select indices
        E = PPMI.make_ppmi_matrix(Y, M.vocab, M.ivocab, T, r)
        @test isapprox(E[indices, :], D[indices, :])
        @test isapprox(E[:, indices], D[:, indices])
        Ẽ = PPMI.make_ppmi_matrix(Y, δX, M.vocab, M.ivocab, T, r)
        @test isapprox(Ẽ[indices, :], D̃[indices, :], rtol=0.001, atol=0.01)
        @test !isapprox(E[indices, :], D̃[indices, :], rtol=0.001, atol=0.01)
        @test isapprox(Ẽ[:, indices], D̃[:, indices], rtol=0.001, atol=0.01)
        @test !isapprox(E[:, indices], D̃[:, indices], rtol=0.001, atol=0.01)
    end

end
