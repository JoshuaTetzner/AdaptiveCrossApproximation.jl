abstract type PivStrat end
abstract type PivStratFunctor end

abstract type GeoPivStrat <: PivStrat end
abstract type ValuePivStrat <: PivStrat end
abstract type ConvPivStrat <: PivStrat end

abstract type GeoPivStratFunctor <: PivStratFunctor end
abstract type ConvPivStratFunctor <: PivStratFunctor end
abstract type ValuePivStratFunctor <: PivStratFunctor end
