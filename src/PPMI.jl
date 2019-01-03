module PPMI

using SparseArrays


# Return the total number of tokens (i.e. the "in-vocab" word count)
function sum_counts(vocab)
    tot = 0
    for word_info in values(vocab)
        tot += word_info.count
    end
    return tot
end

# Co-occ "mass" per occurence
function cooc_ratio(window)
    return 2 * sum(1.0/n for n in 1:window)
end


function make_ppmi_matrix(X::SparseMatrixCSC, vocab, ivocab, T, r)
    V = length(ivocab)
    Z = nnz(X)
    Is = zeros(Int64, Z)
    Js = zeros(Int64, Z)
    PPMIs = zeros(Z)
    for (k, (i, j, Xij)) in enumerate(zip(findnz(X)...))
        Is[k] = i; Js[k] = j;
        count_i = vocab[ivocab[i]].count
        count_j = vocab[ivocab[j]].count
        pmi = log(Xij * T / (r * count_i * count_j))
        ppmi = max(0, pmi)
        PPMIs[k] = ppmi
    end
    return sparse(Is, Js, PPMIs, V, V)
end


function make_ppmi_matrix(X::SparseMatrixCSC, δX::SparseMatrixCSC, vocab, ivocab, T, r)
    V = length(ivocab)
    Z = nnz(X)
    Is = zeros(Int64, Z)
    Js = zeros(Int64, Z)
    PPMIs = zeros(Z)
    for (k, (i, j, Xij)) in enumerate(zip(findnz(X)...))
        Is[k] = i; Js[k] = j;
        count_i = vocab[ivocab[i]].count
        count_j = vocab[ivocab[j]].count
        if δX[i, j] != 0.0
            count_i = max(1, count_i - sum(δX[:, i]) / r)
            count_j = max(1, count_j - sum(δX[:, j]) / r)
            # if (Xij < δX[i, j])
            #     # This seems to just be small rounding errors
            #     println("$i - $(ivocab[i]), $j - $(ivocab[j]) : $Xij < $(δX[i, j])")
            # end
            Xij = max(0, Xij - δX[i, j])
        end
        pmi = log(Xij * T / (r * count_i * count_j))
        ppmi = max(0, pmi)
        PPMIs[k] = ppmi
    end
    return sparse(Is, Js, PPMIs, V, V)
end

end
