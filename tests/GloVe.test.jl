using Test
using SparseArrays
using ForwardDiff

@testset "GloVe" begin

    include("../src/GloVe.jl")
    test_corpus_path = "test_corpus.txt"
    test_cooc_path = "test_cooc.bin"
    embedding_path = "../embeddings/vectors-C0-V20-W8-D25-R0.05-E15-S1.bin"
    full_cooc_path = "../embeddings/cooc-C0-V20-W8.bin"

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
        X = GloVe.load_cooc(test_cooc_path, M.V)
        X_1 = GloVe.docs2cooc(eachline(test_corpus_path), M.vocab, M.d)
        @test isapprox(X, X_1)
        X_2 = spzeros(M.V, M.V)
        for line in eachline(test_corpus_path)
            X_2 += GloVe.doc2cooc(line, M.vocab, M.d)
        end
        @test isapprox(X, X_2)
        test_words = ["april", "the", "and", "having"]
        indices = [M.vocab[word].index for word in test_words]
        X_3 = GloVe.load_cooc(test_cooc_path, M.V, indices)
        @test X_3[indices, :] == X[indices, :]
        @test X_3[:, indices] == X[:, indices]
    end

    @testset "Cooc Matrix saving/loading" begin
        V = 100
        test_path = "test_cooc_2.tmp"
        X = sprand(V, V, 0.1)
        GloVe.save_coocs(test_path, X)
        X_1 = GloVe.load_cooc(test_path, V)
        @test isapprox(X, X_1)
        rm(test_path)
    end

    @testset "Gradients and Hessians" begin
        ∇ = ForwardDiff.gradient
        ∇² = ForwardDiff.hessian

        # Make some very toy examples
        (V, D) = (25, 4)
        (W, b_w, U, b_u) = (rand(V, D), rand(V), rand(V, D), rand(V))
        X = sprand(V, V, 0.5); X += X'

        ∇J = ∇(w->GloVe.J(w, b_w, U, b_u, X), W)
        ∇²J = ∇²(w->GloVe.J(w, b_w, U, b_u, X), W)

        # Convert ForwardDiff's indexing
        reindex(i) = V  * collect(0:D-1) .+ i

        gradients_match = true
        hessians_match = true

        for i in 1:V
            gradients_match &= isapprox(∇J[i,:], GloVe.∇Li(W, b_w, U, b_u, X, i))
            hessians_match &= isapprox(∇²J[reindex(i), reindex(i)], GloVe.∇²Li(U, X, i))
        end

        @test gradients_match
        @test hessians_match

    end

    @testset "Influence Function approximation" begin
        document = "the quick brown fox jumped over the lazy dog"
        X = GloVe.load_cooc(full_cooc_path, M.V)
        deltas = GloVe.compute_IF_deltas(document, M, X)
        # Check only words in document are affected
        @test Set(keys(deltas)) == Set([M.vocab[word].index for word in split(document)])
        all_non_zeros = true
        for (idx, delta) in pairs(deltas)
            all_non_zeros &= (length(delta) == M.D) && (delta != zeros(M.D))
        end
        @test all_non_zeros

        # Now test more efficient version
        target_words = split("jumped dog excluded")
        target_indices = [M.vocab[word].index for word in target_words]
        H = GloVe.inv_hessians_for(target_indices, M, X)
        G = GloVe.gradients_for(target_indices, M, X)

        select_deltas = GloVe.compute_IF_deltas(document, M, X, target_indices, H, G)
        @test length(select_deltas) == 2  # for "jumped" and "dog"

        same_values = true
        for (idx, delta) in pairs(select_deltas)
            same_values &= delta == deltas[idx]
        end
        @test same_values
    end


end
