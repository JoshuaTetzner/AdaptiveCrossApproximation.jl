# Extracts per-basis-function edge directions and face normals from a BEAST
# Raviart–Thomas space, for direction-aware tree-mimicry pivoting.

@inline _unit(v) = v / norm(v)
@inline _cleanzero(x) = iszero(x) ? zero(x) : x
@inline _clean(v) = typeof(v)(_cleanzero(v[1]), _cleanzero(v[2]), _cleanzero(v[3]))

@inline function _canonicalsign(v)
    v = _clean(v)
    return if v[1] < 0 || (v[1] == 0 && v[2] < 0) || (v[1] == 0 && v[2] == 0 && v[3] < 0)
        _clean(-v)
    else
        v
    end
end

@inline function _edgeverts(cell, refid)
    refid == 1 && return cell[2], cell[3]
    refid == 2 && return cell[3], cell[1]
    return cell[1], cell[2]
end

@inline function _cellnormal(verts, cell)
    return cross(verts[cell[2]] - verts[cell[1]], verts[cell[3]] - verts[cell[1]])
end

function AdaptiveCrossApproximation.rwgorientations(rwg::BEAST.Space)
    mesh = rwg.geo
    verts = vertices(mesh)
    meshcells = collect(cells(mesh))
    nc = length(meshcells)

    ℓ = similar(rwg.pos)
    n = similar(rwg.pos)
    cn = similar(rwg.pos, nc)

    @inbounds for c in 1:nc
        cn[c] = _clean(_unit(_cellnormal(verts, meshcells[c])))
    end

    @inbounds for i in eachindex(rwg.fns)
        fn = rwg.fns[i]
        sh = fn[1]

        cell = meshcells[sh.cellid]
        a, b = _edgeverts(cell, sh.refid)

        ℓ[i] = _canonicalsign(_unit(verts[b] - verts[a]))

        ni = cn[fn[1].cellid]
        if length(fn) > 1
            nj = cn[fn[2].cellid]
            if abs(dot(ni, nj)) < 0.9
                n[i] = zero(ni)
                continue
            end
            ni += dot(ni, nj) < 0 ? -nj : nj
        end
        n[i] = _clean(_unit(ni))
    end

    return ℓ, n
end

"""
    EFIEDirectionalFilter(candidatespace, candidatetree)

Build an [`EFIEDirectionalFilter`](@ref) from a Raviart–Thomas candidate space and
its tree. Computes the candidate edge/normal groups and the per-node dominant
normal-direction sets used by the filter.
"""
function AdaptiveCrossApproximation.EFIEDirectionalFilter(
    candidatespace::BEAST.Space, candidatetree
)
    candidateedges, candidatenormals = AdaptiveCrossApproximation.rwgorientations(
        candidatespace
    )
    edgeids, _, _, _ = AdaptiveCrossApproximation.basisfunction_orientation_ids(
        candidateedges, candidatenormals
    )
    nodedirectionids, basisdirections, _ = AdaptiveCrossApproximation.node_normal_orientation_sets(
        candidatenormals, candidatetree
    )
    return AdaptiveCrossApproximation.EFIEDirectionalFilter(
        edgeids, basisdirections, nodedirectionids
    )
end
