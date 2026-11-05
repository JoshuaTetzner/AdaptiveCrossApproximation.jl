nextrc!(buf, A::BEASTKernelMatrix, i, j) = A(buf, i, j)
nextrc!(buf, A::BEASTKernelMatrix, i::Integer, j) = A(buf, i:i, j)
nextrc!(buf, A::BEASTKernelMatrix, i, j::Integer) = A(buf, i, j:j)
nextrc!(buf, A::BEASTKernelMatrix, i::Integer, j::Integer) = A(buf, i:i, j:j)
