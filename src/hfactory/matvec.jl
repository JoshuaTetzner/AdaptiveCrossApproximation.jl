# The scratch holds the rank-sized intermediate product.
function HFactory.matvecscratch(
    blocks::AbstractVector{<:LowRankMatrix}, ::Type{S}
) where {S}
    return Vector{S}(undef, maximum(rank, blocks; init=0))
end

function HFactory.blockmul!(y, B::LowRankMatrix, x, α, β, z::Vector, ::Val{:N})
    Vx = mul!(view(z, 1:rank(B)), B.V, x)
    return mul!(y, B.U, Vx, α, β)
end

function HFactory.blockmul!(y, B::LowRankMatrix, x, α, β, z::Vector, ::Val{:T})
    Ux = mul!(view(z, 1:rank(B)), transpose(B.U), x)
    return mul!(y, transpose(B.V), Ux, α, β)
end

function HFactory.blockmul!(y, B::LowRankMatrix, x, α, β, z::Vector, ::Val{:C})
    Ux = mul!(view(z, 1:rank(B)), adjoint(B.U), x)
    return mul!(y, adjoint(B.V), Ux, α, β)
end
