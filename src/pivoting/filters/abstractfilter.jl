# Pivot filters restrict which tree nodes and basis-function indices a pivoting
# strategy may descend into and select. A filter is a small immutable descriptor
# (`PivotingFilter`) paired with a mutable per-factorization state
# (`PivotingFilterState`) that the strategy threads through the pivot-selection
# hooks below. The identity hooks defined here accept every node/index and run a
# single directionless phase, so a concrete filter only overrides the hooks it
# actually needs. `NoFilter` is exactly that identity and keeps the plain
# (unfiltered) pivoting behaviour.

"""
    PivotingFilter

Abstract supertype for pivot filters. A concrete filter is an immutable descriptor
holding the data that restricts pivot selection (e.g. per-index orientation ids);
it is paired with a [`PivotingFilterState`](@ref) built once per factorization.

Currently consumed only by [`TreeMimicryPivoting`](@ref); the interface hooks are
written against the abstract [`PivStratFunctor`](@ref) so they can later be
generalized to other strategies.

See also [`NoFilter`](@ref), [`EFIEDirectionalFilter`](@ref).
"""
abstract type PivotingFilter end

"""
    PivotingFilterState

Abstract supertype for the mutable per-factorization state of a
[`PivotingFilter`](@ref). The identity hooks defined on this type make an
un-overridden filter behave like [`NoFilter`](@ref): every node/index is accepted,
no edge weighting is applied, and a single directionless phase runs. A concrete
filter overrides only the hooks whose behaviour differs from the identity.
"""
abstract type PivotingFilterState end

"""
    NoFilter

Identity [`PivotingFilter`](@ref): accepts every node and index and runs a single
directionless phase, recovering the plain (unfiltered) pivoting behaviour. It is
the default filter of [`TreeMimicryPivoting`](@ref).
"""
struct NoFilter <: PivotingFilter end

"""
    NoFilterState

Per-factorization state of [`NoFilter`](@ref); stateless. Inherits every hook from
the [`PivotingFilterState`](@ref) identity defaults.
"""
struct NoFilterState <: PivotingFilterState end

# --- PivotingFilter interface (identity defaults) --------------------------
# Concrete filters override the subset of hooks they need.
@inline _reset_filterstate!(::PivotingFilterState) = nothing
@inline _accepts_node(::PivotingFilterState, ::PivStratFunctor, ::Int, ::Int32) = true
@inline _accepts_index(::PivotingFilterState, ::PivStratFunctor, ::Int, ::Int32) = true
@inline _edge_count(::PivotingFilterState, ::PivStratFunctor, ::Int, ::Int) = 0
@inline _reset_phase!(::PivotingFilterState, ::Int32) = nothing
@inline _initial_direction(::PivotingFilterState, ::PivStratFunctor, ::Int) = _nodirection()
@inline _advance_direction(::PivotingFilterState, ::PivStratFunctor, ::Int, ::Int32) =
    (_nodirection(), false)

"""
    _filterstate(filter::PivotingFilter, convcrit) -> PivotingFilterState

Build the per-factorization [`PivotingFilterState`](@ref) for `filter`, given the
factorization's convergence criterion `convcrit`. Direction-phased filters consult
`convcrit` to decide when a phase is finished and therefore store it in their
state; such filters provide a method. [`NoFilter`](@ref) needs no convergence
criterion and is built directly, so it does not implement this hook.
"""
function _filterstate end
