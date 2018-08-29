using Test
using SparseArrays

@testset "GloVe" begin

    include("../src/GloVe.jl")
    corpus_path = "test_corpus.txt"
    cooc_path = "test_cooc.bin"
    embedding_path = "../embeddings/vectors-C0-V20-W8-D25-R0.05-E15-S1.bin"

    @testset "Model loading" begin
        global M = GloVe.load_model(embedding_path)
        @test M.D == 25 # vector dim
        @test M.d == 8 # window
        # Test against some random examples from `head vocab.txt`
        @test M.ivocab[3] == "and"
        @test M.vocab["is"] == GloVe.WORD_INFO(6, 213234)
    end

    @testset "Vector saving" begin
        vec_path = "test_vecs.tmp"
        (V, D) = (100, 10)
        (W, b_w, U, b_u) = (rand(V, D), rand(V), rand(V, D), rand(V))
        GloVe.save_bin_vectors(vec_path, W, b_w, U, b_u)
        (W_1, b_w_1, U_1, b_u_1) = GloVe.load_bin_vectors(vec_path, V)
        @test isapprox(W, W_1)
        @test isapprox(b_w, b_w_1)
        @test isapprox(U, U_1)
        @test isapprox(b_u, b_u_1)
        rm(vec_path)
    end

    @testset "Cooc Matrix parsing/creation" begin
        X = GloVe.load_cooc(cooc_path, M.V)
        X_1 = GloVe.docs2cooc(eachline(corpus_path), M.vocab, M.d)
        @test isapprox(X, X_1)
        X_2 = spzeros(M.V, M.V)
        for line in eachline(corpus_path)
            X_2 += GloVe.doc2cooc(line, M.vocab, M.d)
        end
        @test isapprox(X, X_2)
        test_words = ["april", "the", "and", "having"]
        indices = [M.vocab[word].index for word in test_words]
        X_3 = GloVe.load_cooc(cooc_path, M.V, indices)
        @test X_3[indices, :] == X[indices, :]
        @test X_3[:, indices] == X[:, indices]
    end

    @testset "Cooc Matrix saving/loading" begin
        V = 100
        test_path = "cooc_test_2.tmp"
        X = sprand(V, V, 0.1)
        GloVe.save_coocs(test_path, X)
        X_1 = GloVe.load_cooc(test_path, V)
        @test isapprox(X, X_1)
        rm(test_path)
    end

end
