using Polynomials

mutable struct OversampIFNormEst{F} <: ConvCrit
    tol::F
end

mutable struct OversampIFNormEstFunctor{F} <: ConvCritFunctor
    normUV::F
    lastnorms::Vector{F}
    normdirections::Vector{Int32}
    tol::F
    lastconverged::Bool
    currentdirection::Int32
end

function (cc::OversampIFNormEst{F})() where {F}
    return cc(0)
end

function (cc::OversampIFNormEst{F})(maxrank::Int) where {F}
    return OversampIFNormEstFunctor(
        F(0), zeros(F, maxrank), zeros(Int32, maxrank), cc.tol, false, typemin(Int32)
    )
end

function reset!(convcrit::OversampIFNormEstFunctor)
    convcrit.normUV = zero(convcrit.normUV)
    fill!(convcrit.lastnorms, zero(eltype(convcrit.lastnorms)))
    fill!(convcrit.normdirections, Int32(0))
    convcrit.lastconverged = false
    convcrit.currentdirection = typemin(Int32)
    return nothing
end

tolerance(cc::OversampIFNormEstFunctor) = cc.tol
tolerance(cc::OversampIFNormEst) = cc.tol

function normF!(
    convcrit::OversampIFNormEstFunctor, rcbuffer::AbstractVector{K}, npivot::Int
) where {K}
    convcrit.normUV = ((npivot - 1) * convcrit.normUV + norm(rcbuffer)) / npivot
    return nothing
end

function reset_phase!(convcrit::OversampIFNormEstFunctor, direction::Integer, npivot::Int)
    convcrit.currentdirection = Int32(direction)
    return nothing
end

function _extrapolated_converged(
    convcrit::OversampIFNormEstFunctor{F}, npivot::Int
) where {F}
    nfit = npivot - 1
    nfit < 3 && return true

    x = Vector{F}(undef, nfit)
    y = sort!(copy(view(convcrit.lastnorms, 1:nfit)); rev=true)
    @inbounds for i in 1:nfit
        x[i] = F(i)
        y[i] = log10(y[i])
    end

    f2 = fit(x, y, 2)
    #println(f2(F(nfit + 1)), " <= ", log10(tolerance(convcrit) * convcrit.normUV))
    return f2(F(nfit + 1)) <= log10(tolerance(convcrit) * convcrit.normUV)
end

function _direction_extrapolated_converged(
    convcrit::OversampIFNormEstFunctor{F}, npivot::Int
) where {F}
    dir = convcrit.currentdirection
    dir <= 0 && return true
    nfit = 0
    @inbounds for i in 1:(npivot - 1)
        nfit += convcrit.normdirections[i] == dir
    end
    nfit < 3 && return true

    x = Vector{F}(undef, nfit)
    y = Vector{F}(undef, nfit)
    j = 0
    @inbounds for i in 1:(npivot - 1)
        convcrit.normdirections[i] == dir || continue
        j += 1
        x[j] = F(j)
        y[j] = log10(convcrit.lastnorms[i])
    end

    f2 = fit(x, y, 2)
    #println(f2(F(nfit + 1)), " <= ", log10(tolerance(convcrit) * convcrit.normUV))
    return f2(F(nfit + 1)) <= log10(tolerance(convcrit) * convcrit.normUV)
end

function (convcrit::OversampIFNormEstFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    rcnorm = norm(rcbuffer)
    isapprox(rcnorm, 0.0) && return npivot - 1, false
    length(convcrit.lastnorms) < npivot && resize!(convcrit.lastnorms, npivot)
    length(convcrit.normdirections) < npivot && resize!(convcrit.normdirections, npivot)
    convcrit.lastnorms[npivot] = rcnorm
    convcrit.normdirections[npivot] = convcrit.currentdirection
    converged =
        rcnorm <= tolerance(convcrit) * convcrit.normUV &&
        _extrapolated_converged(convcrit, npivot) &&
        _direction_extrapolated_converged(convcrit, npivot)
    keepgoing = !(converged && convcrit.lastconverged)
    convcrit.lastconverged = converged
    #println(rcnorm, " <= ", tolerance(convcrit) * convcrit.normUV)
    return npivot, keepgoing
end
