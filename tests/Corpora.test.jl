using Test

@testset "Corpora" begin

    include("../src/Corpora.jl")
    wiki_path = "../corpora/simplewikiselect"

    @testset "Corpus constructor" begin
        global c = Corpora.Corpus(wiki_path)
        @test c.corpus_path == abspath("$wiki_path.txt")
        @test c.meta_path == abspath("../corpora/simplewikiselect.meta.json")
    end

    @testset "Document retrival" begin
        target = 11
        line = ""
        open(c.corpus_path) do f
            for i in 1:target
                line = readline(f)
            end
        end
        @test line == Corpora.get_text(c, target)

        open(c.corpus_path) do f
            @test line == Corpora.get_text(f, c, target)
        end
    end


end
