using Test

@testset "GloVe" begin

    include("../src/GloVe.jl")
    corpus_path = "test_corpus.txt"
    cooc_path = "test_cooc.bin"
    embedding_path = "../embeddings/vectors-C0-V20-W8-D25-R0.05-E15-S1.bin"

    @testset "Model loading" begin
        global M = GloVe.load_model(embedding_path)
        @test M.D == 25
        @test M.d == 8
        # Test against some random examples from `head vocab.txt`
        @test M.ivocab[3] == "and"
        @test M.vocab["is"] == GloVe.WORD_INFO(6, 213234)
    end

    @testset "Coocurence Matrix" begin
        X = GloVe.load_cooc(cooc_path, M.V)
        X_2 = GloVe.docs2cooc(eachline(corpus_path), M.vocab, M.d)
        @test isapprox(X, X_2)
    end

end
