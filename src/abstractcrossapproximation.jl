abstract type CrossApproximation end

nextrc!(buf, M::AbstractArray{K}, i, j) where {K} = (buf .= view(M, i, j))
