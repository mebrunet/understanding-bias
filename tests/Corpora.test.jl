using Test

@testset "Corpora" begin

    include("../src/Corpora.jl")
    wiki_path = "../corpora/simplewikiselect"
    nyt_path = "../corpora/nytselect"

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

        indices = [15, 100, 789]
        texts = [Corpora.get_text(c, idx) for idx in indices]
        @test texts == Corpora.get_texts(c, indices)
    end

    if isfile("$nyt_path.txt")
        @testset "NYT corpus" begin
            nyt = Corpora.Corpus(nyt_path)
            target = 101
            line = ""
            open(nyt.corpus_path) do f
                for i in 1:target
                    line = readline(f)
                end
            end
            @test line == Corpora.get_text(nyt, target)
        end
    end


end
