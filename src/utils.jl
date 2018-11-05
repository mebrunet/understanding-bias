# Handy helpers

function include_everywhere(filepath)
    include(filepath) # Load on Node 1 first, triggering any precompile
    if nprocs() > 1
        fullpath = joinpath(@__DIR__, filepath)
        @sync for p in workers()
            @async remotecall_wait(include, p, fullpath)
        end
    end
end


function zip_word_sets(U, V)
    zipped = ""
    for (u, v) in zip(U, V)
        zipped *= u * " " * v * " "
    end
    return zipped
end


# Make a bunch of test "documents" filled with WEAT words
function make_test_docs(weat_word_set)
    SA = zip_word_sets(weat_word_set.S, weat_word_set.A)
    SB = zip_word_sets(weat_word_set.S, weat_word_set.B)
    TA = zip_word_sets(weat_word_set.T, weat_word_set.A)
    TB = zip_word_sets(weat_word_set.T, weat_word_set.B)
    return (SA=SA, SB=SB, TA=TA, TB=TB)
end


# Regex Helper
function extract(string, regex; trim=(0,0), cast=nothing)
    m = match(regex, string)
    tmp = nothing
    if (m != nothing)
        tmp = m.match[1+trim[1]:end-trim[2]]
        if (cast != nothing)
            tmp = parse(cast, tmp)
        end
    end
    return tmp
end


# Get file info from naming convention
function fileinfo(filepath)
    corpus = extract(filepath, r"C[0-9]+", trim=(1,0), cast=Int64)
    min_vocab = extract(filepath, r"V[0-9]+", trim=(1,0), cast=Int64)
    window = extract(filepath, r"W[0-9]+", trim=(1,0), cast=Int64)
    dimension = extract(filepath, r"D[0-9]+", trim=(1,0), cast=Int64)
    eta = extract(filepath, r"R[0-9]+.[0-9]+", trim=(1,0), cast=Float64)
    max_iters = extract(filepath, r"E[0-9]+", trim=(1,0), cast=Int64)
    seed = extract(filepath, r"S[0-9]+", trim=(1,0), cast=Int64)
    tmp = extract(filepath, r".[0-9]{3}.bin$", trim=(1, 4), cast=Int64)
    iters = tmp == nothing ? max_iters : tmp
    return (corpus=corpus, min_vocab=min_vocab, window=window, dimension=dimension,
            eta=eta, max_iters=max_iters, seed=seed, iters=iters)
end
