# Orientation utilities for direction-aware tree-mimicry pivoting.
#
# These helpers turn per-basis-function geometric data (edge directions, face
# normals) into the integer group IDs and per-node direction sets consumed by
# the `EFIEDirectionalFilter`. They depend only on the ACA tree stubs
# (`values`, `children`, `parent`) and Julia Base, so backend-specific data
# extraction (e.g. `rwgorientations` for BEAST spaces) is provided by extensions.

"""
    rwgorientations(space)

Compute per-basis-function edge directions and face normals for `space`.

Returns `(edges, normals)`, one unit vector each per basis function, suitable as
input to [`basisfunction_orientation_ids`](@ref) and
[`node_normal_orientation_sets`](@ref).

Implemented in the `ACABEAST` extension for BEAST Raviart–Thomas spaces.
"""
function rwgorientations(space)
    return error(
        "rwgorientations is not implemented for $(typeof(space)). Load BEAST for Raviart–Thomas spaces.",
    )
end

"""
    orientation_ids(q)

Assign a sequential `Int32` id to each distinct value in `q`, in order of first
occurrence. Every entry gets an id (starting at `1`); use [`normal_orientation_ids`](@ref)
instead if zero-valued keys should be exempt (id `0`).

# Returns

  - `(ids, nids)`: per-entry group ids and the number of distinct groups found.
"""
function orientation_ids(q)
    key_to_id = Dict{eltype(q),Int32}()
    ids = Vector{Int32}(undef, length(q))
    nextid = Int32(0)

    @inbounds for i in eachindex(q)
        id = get(key_to_id, q[i], Int32(0))
        if id == 0
            nextid += 1
            id = nextid
            key_to_id[q[i]] = id
        end
        ids[i] = id
    end

    return ids, Int(nextid)
end

@inline function _iszero_direction_key(key)
    return iszero(key[1]) && iszero(key[2]) && iszero(key[3])
end

"""
    normal_orientation_ids(q)

Like [`orientation_ids`](@ref), but zero-valued keys (i.e. "no normal") are left at
id `0` instead of starting a new group.

# Returns

  - `(ids, nids)`: per-entry group ids (`0` for zero keys) and the number of
    distinct non-zero groups found.
"""
function normal_orientation_ids(q)
    key_to_id = Dict{eltype(q),Int32}()
    ids = zeros(Int32, length(q))
    nextid = Int32(0)

    @inbounds for i in eachindex(q)
        _iszero_direction_key(q[i]) && continue
        id = get(key_to_id, q[i], Int32(0))
        if id == 0
            nextid += 1
            id = nextid
            key_to_id[q[i]] = id
        end
        ids[i] = id
    end

    return ids, Int(nextid)
end

@inline function _canonical_direction_key(v; digits=1)
    scale = 10.0^digits
    x = round(Int16, abs(v[1]) * scale)
    y = round(Int16, abs(v[2]) * scale)
    z = round(Int16, abs(v[3]) * scale)
    return (x, y, z)
end

@inline function _keyvector(key)
    v = (Float64(key[1]), Float64(key[2]), Float64(key[3]))
    nv = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    iszero(nv) && return (0.0, 0.0, 0.0)
    return (v[1] / nv, v[2] / nv, v[3] / nv)
end

@inline function _absdot(a, b)
    return abs(a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
end

"""
    basisfunction_orientation_ids(edges, normals; edge_digits=1, normal_digits=1)

Assign integer orientation-group ids to a set of basis functions from their edge
directions and face normals.

# Arguments

  - `edges`: one direction vector per basis function (e.g. RWG edge directions)
  - `normals`: one normal vector per basis function; zero vectors are treated as
    "no normal" and get id `0`

# Returns

  - `(edgeids, normalids, nedgeids, nnormalids)`: per-basis-function group ids for
    edges and normals (both `Vector{Int32}`), plus the number of distinct groups
    found for each. Directions that are antiparallel (up to rounding at `digits`
    decimal places) are assigned the same id, since only the underlying line
    matters for orientation grouping.

# See also

[`node_normal_orientation_sets`](@ref), [`orientation_ids`](@ref),
[`normal_orientation_ids`](@ref)
"""
function basisfunction_orientation_ids(edges, normals; edge_digits=1, normal_digits=1)
    edge_keys = [_canonical_direction_key(edge; digits=edge_digits) for edge in edges]
    normal_keys = [
        _canonical_direction_key(normal; digits=normal_digits) for normal in normals
    ]

    edgeids, nedgeids = orientation_ids(edge_keys)
    normalids, nnormalids = normal_orientation_ids(normal_keys)

    return edgeids, normalids, nedgeids, nnormalids
end

function _key_representatives(keys, ids, nids::Integer)
    representatives = Vector{eltype(keys)}(undef, nids)
    @inbounds for i in eachindex(keys)
        id = ids[i]
        iszero(id) && continue
        representatives[id] = keys[i]
    end
    return representatives
end

function node_normal_orientation_sets(
    normalids::AbstractVector{Int32},
    tree,
    nnormalids::Integer,
    representatives;
    max_orientations::Int=3,
    primary_probability::Float64=0.2,
    secondary_probability::Float64=0.1,
    active_probability::Float64=0.7,
    orth_tol::Float64=0.25,
)
    out = Vector{Vector{Int32}}(undef, length(tree.nodes))

    counts = zeros(Int, nnormalids)
    touched = Int32[]
    selected = Int32[]
    inherited = Int32[]

    function visit!(node::Int, parentclasses::AbstractVector{Int32})
        empty!(touched)
        empty!(selected)
        empty!(inherited)
        idcs = values(tree, node)
        ntotal = length(idcs)

        @inbounds for i in idcs
            id = normalids[i]
            iszero(id) && continue
            if iszero(counts[id])
                push!(touched, id)
            end
            counts[id] += 1
        end

        sort!(touched; by=id -> counts[id], rev=true)
        selectedcount = 0
        if !isempty(touched) && counts[first(touched)] / ntotal >= primary_probability
            firstid = first(touched)
            push!(selected, firstid)
            selectedcount += counts[firstid]

            @inbounds for k in 2:length(touched)
                id = touched[k]
                counts[id] / ntotal < secondary_probability && break
                v = _keyvector(representatives[id])
                orthogonal = true
                for sid in selected
                    u = _keyvector(representatives[sid])
                    if _absdot(v, u) > orth_tol
                        orthogonal = false
                        break
                    end
                end
                if orthogonal
                    push!(selected, id)
                    selectedcount += counts[id]
                    length(selected) == max_orientations && break
                end
            end
        end

        if length(selected) >= 2 && selectedcount / ntotal >= active_probability
            out[node] = copy(selected)
        elseif !isempty(parentclasses)
            @inbounds for id in parentclasses
                id <= nnormalids && counts[id] > 0 && push!(inherited, id)
            end
            out[node] = copy(inherited)
        else
            out[node] = Int32[]
        end

        @inbounds for id in touched
            counts[id] = 0
        end

        for child in children(tree, node)
            visit!(Int(child), out[node])
        end
    end

    for node in eachindex(tree.nodes)
        parent(tree, node) == 0 && visit!(Int(node), Int32[])
    end

    return out
end

"""
    node_normal_orientation_sets(normals, tree; normal_digits=1, max_orientations=3,
        primary_probability=0.2, secondary_probability=0.1, active_probability=0.7,
        orth_tol=0.25)

Determine, per tree node, which basis-function normal directions dominate its subtree.

Walks `tree` top-down (a node must implement `values`, `children`, `parent`; see
[`TreeMimicryPivoting`](@ref)). At each node it counts how often each normal-direction
group occurs among its basis functions and selects up to `max_orientations` mutually
(near-)orthogonal directions (via `orth_tol` on `abs(dot(·,·))`) that together explain
at least `active_probability` of the node's basis functions, requiring the dominant
direction to cover at least `primary_probability` and secondary ones at least
`secondary_probability`. If no direction set is dominant enough, the node inherits its
parent's set. Used to drive the [`EFIEDirectionalFilter`](@ref), which restricts
descent/pivoting to one orientation group at a time.

# Returns

  - `(nodesets, normalids, nnormalids)`: `nodesets[node]::Vector{Int32}` is the set of
    dominant direction ids for `node`; `normalids`/`nnormalids` are as returned by
    [`normal_orientation_ids`](@ref) on `normals`.

# See also

[`basisfunction_orientation_ids`](@ref), [`EFIEDirectionalFilter`](@ref)
"""
function node_normal_orientation_sets(
    normals,
    tree;
    normal_digits=1,
    max_orientations=3,
    primary_probability=0.2,
    secondary_probability=0.1,
    active_probability=0.7,
    orth_tol=0.25,
)
    normal_keys = [
        _canonical_direction_key(normal; digits=normal_digits) for normal in normals
    ]
    normalids, nnormalids = normal_orientation_ids(normal_keys)
    representatives = _key_representatives(normal_keys, normalids, nnormalids)

    nodesets = node_normal_orientation_sets(
        normalids,
        tree,
        nnormalids,
        representatives;
        max_orientations=max_orientations,
        primary_probability=primary_probability,
        secondary_probability=secondary_probability,
        active_probability=active_probability,
        orth_tol=orth_tol,
    )

    return nodesets, normalids, nnormalids
end
