# Single entries of a BEAST matrix, such as the random samples of the convergence check.
# Through the block assembler, every entry would reduce the assembly data to its elements
# and lock the scratch pool. Here an entry is summed directly over the element integrals of
# the shapes of its test and trial function. This needs the element-pair `integrate!` of
# BEAST > 2.10; older versions keep the generic path.
if isdefined(BEAST, :ApplyIntegrate)
    function assembleentries_body!(
        out,
        biop,
        test_functions,
        test_elements,
        test_shapes,
        trial_functions,
        trial_elements,
        trial_shapes,
        zlocal,
        quadrature_data,
        entries;
        quadstrat,
    )
        qbuffer = BEAST.quadraturebuffer(quadstrat, test_functions, trial_functions)
        qaction = BEAST.ApplyIntegrate(qbuffer)
        for (k, (m, n)) in enumerate(entries)
            value = zero(eltype(out))
            for s in test_functions.fns[m], t in trial_functions.fns[n]
                p, q = s.cellid, t.cellid
                fill!(zlocal, 0)
                BEAST.integrate!(
                    biop,
                    test_shapes,
                    trial_shapes,
                    p,
                    test_elements[p],
                    q,
                    trial_elements[q],
                    quadrature_data,
                    quadstrat,
                    zlocal,
                    test_functions,
                    p,
                    trial_functions,
                    q;
                    action=qaction,
                )
                value += s.coeff * zlocal[s.refid, t.refid] * t.coeff
            end
            out[k] = value
        end
        return out
    end

    function AdaptiveCrossApproximation._fillrest!(
        rest::AbstractVector,
        indices::Vector{Tuple{Int,Int}},
        K::AdaptiveCrossApproximation.BEASTKernelMatrix{T,<:BEAST.AssembleblockbodyFunctor},
        rowidcs::AbstractArray{Int},
        colidcs::AbstractArray{Int},
        nactive::Int,
    ) where {T}
        length(rest) < nactive && resize!(rest, nactive)
        a = K.nearassembler
        entries = ((rowidcs[r], colidcs[c]) for (r, c) in view(indices, 1:nactive))
        assembleentries_body!(
            rest,
            a.biop,
            a.tfs,
            a.testelements,
            BEAST.refspace(a.tfs),
            a.bfs,
            a.trialelements,
            BEAST.refspace(a.bfs),
            _localzlocal(T, a),
            a.quadraturedata,
            entries;
            quadstrat=a.quadstrat,
        )
        return nothing
    end
end
