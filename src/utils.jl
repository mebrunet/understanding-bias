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
