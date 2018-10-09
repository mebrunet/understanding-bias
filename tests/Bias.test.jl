@testset "Bias" begin

    include("../src/Bias.jl")
    !isdefined(Main, :GloVe) && include("../src/GloVe.jl")

    embedding_path = "../embeddings/vectors-C0-V20-W8-D25-R0.05-E15-S1.bin"
    global M = GloVe.load_model(embedding_path)

    @testset "Basic functioning" begin
        global weat_idx_set_1 = Bias.get_weat_idx_set(Bias.WEAT_WORD_SETS[1], M.vocab)
        weat_idx_set_2 = Bias.get_weat_idx_set(Bias.WEAT_WORD_SETS[2], M.vocab)
        # Rudimentary, but at least verifies dimensionality
        @test typeof(Bias.effect_size(M.W, weat_idx_set_1)) <: Real
        @test typeof(Bias.effect_size(M.W, weat_idx_set_2)) <: Real
        @test typeof(Bias.p_value(M.W, weat_idx_set_1)) <: Real
        @test typeof(Bias.p_value(M.W, weat_idx_set_2)) <: Real
    end

    @testset "Intution of WEAT" begin
        (δ,D) = (8, 25)
        dir_1 = 10 * rand(δ,D)
        dir_2 = 10 * rand(δ,D)
        S = dir_1 + rand(δ,D)
        T = dir_2 + rand(δ,D)
        A = dir_1 + rand(δ,D)
        B = dir_2 + rand(δ,D)

        @test Bias.effect_size(S, T, A, B) > 0.5
        @test Bias.effect_size(T, S, A, B) < -0.5

    end

    @testset "Masking with deltas" begin
        es = Bias.effect_size(M.W, weat_idx_set_1)
        es1 = Bias.effect_size(M.W, weat_idx_set_1, Dict())
        @test es == es1
        # Changes that should increase the effect size by making S closer to A
        deltas = Dict(weat_idx_set_1.S[1]=>100*ones(M.D), weat_idx_set_1.A[1]=>100*ones(M.D),
                      weat_idx_set_1.S[2]=>100*ones(M.D), weat_idx_set_1.A[2]=>100*ones(M.D),
                      weat_idx_set_1.S[3]=>100*ones(M.D), weat_idx_set_1.A[3]=>100*ones(M.D),
                      weat_idx_set_1.S[4]=>100*ones(M.D), weat_idx_set_1.A[4]=>100*ones(M.D))
        es2 = Bias.effect_size(M.W, weat_idx_set_1, deltas)
        @test es < es2

    end

end
