module Corpora

using JSON


struct Corpus
    corpus_path::String
    meta_path::String
    num_documents::Integer
    num_words::Integer
    fields::Array
    index::Array

    function Corpus(path::String, meta_path=nothing)
        corpus_path = path
        base_path = path
        if (endswith(path, ".txt"))
            base_path = path[1:end-4]
        else
            corpus_path = "$path.txt"
        end

        corpus_path = abspath(corpus_path)
        if (!isfile(corpus_path))
            error("Corpus file: $corpus_path not found")
        end

        if (meta_path == nothing)
            meta_path = abspath("$base_path.meta.json")
        else
            meta_path = abspath(meta_path)
        end

        if (!isfile(meta_path))
            error("Metadata file: $meta_path not found")
        end

        meta = JSON.parsefile(meta_path)
        num_documents = meta["num_documents"]
        num_words = meta["num_words"]
        fields = [x for x in meta["fields"] if (x != "line" && x != "byte")]
        index = meta["index"]
        return new(corpus_path, meta_path, num_documents, num_words, fields,
                   index)
    end
end


function get_text(corpus::Corpus, idx::Integer)
    meta = corpus.index[idx]
    open(corpus.corpus_path) do f
        seek(f, meta["byte"])
        return readline(f)
    end
end


function get_text(io::IO, corpus::Corpus, idx::Integer)
    meta = corpus.index[idx]
    seek(io, meta["byte"])
    return readline(io)
end


function get_texts(corpus::Corpus, indices)
    open(corpus.corpus_path) do f
        return [get_text(f, corpus, idx) for idx in indices]
    end
end

end  # end module
