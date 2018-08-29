@testset "Bias" begin

    include("../src/Bias.jl")
    !isdefined(Main, :GloVe) && include("../src/GloVe.jl")

    embedding_path = "../embeddings/vectors-C0-V20-W8-D25-R0.05-E15-S1.bin"
    global M = GloVe.load_model(embedding_path)

    @testset "Basic functioning" begin
        indices_1 = Bias.get_word_indices(Bias.WORD_SETS[1], M.vocab)
        indices_2 = Bias.get_word_indices(Bias.WORD_SETS[2], M.vocab)
        # Rudimentary, but at least verifies dimensionality
        @test Bias.effect_size(M.W, indices_1) > 0.5
        @test Bias.effect_size(M.W, indices_2) > 0.5
        @test Bias.p_value(M.W, indices_1) < 0.05
        @test Bias.p_value(M.W, indices_2) < 0.05
    end

end
