# API Reference

## Core Types

### Cross Approximation

```@docs
CrossApproximation
ACA
ACAᵀ
iACA
```

### Abstract Types

```@docs
PivStrat
PivStratFunctor
ValuePivStrat
GeoPivStrat
ConvPivStrat
ConvCrit
ConvCritFunctor
```

## Pivoting Strategies

### Value-Based

```@docs
MaximumValue
MaximumValueFunctor
RandomSamplingPivoting
RandomSamplingPivotingFunctor
```

### Geometry-Based

```@docs
FillDistance
FillDistanceFunctor
Leja2
Leja2Functor
MimicryPivoting
MimicryPivotingFunctor
TreeMimicryPivoting
TreeMimicryPivotingFunctor
```

### Combined

```@docs
CombinedPivStrat
CombinedPivStratFunctor
```

## Convergence Criteria

### Norm Estimation

```@docs
FNormEstimator
FNormEstimatorFunctor
iFNormEstimator
iFNormEstimatorFunctor
```

### Other Criteria

```@docs
Extrapolation
ExtrapolationFunctor
RandomSamplingConv
RandomSamplingConvFunctor
CombinedConvCrit
CombinedConvCritFunctor
```

## Functions

### Main Functions

```@docs
aca
acaᵀ
nextrc!
```

### Helper Functions

```@docs
leja2!
normF!
tolerance
```