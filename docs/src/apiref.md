# API Reference

## Types
### Abstract Types

```@docs
AdaptiveCrossApproximation.PivStrat
AdaptiveCrossApproximation.PivStratFunctor
AdaptiveCrossApproximation.ValuePivStrat
AdaptiveCrossApproximation.ValuePivStratFunctor
AdaptiveCrossApproximation.GeoPivStrat
AdaptiveCrossApproximation.GeoPivStratFunctor
AdaptiveCrossApproximation.ConvPivStrat
AdaptiveCrossApproximation.ConvPivStratFunctor
AdaptiveCrossApproximation.ConvCrit
AdaptiveCrossApproximation.ConvCritFunctor
```
### Concrete Types
#### Cross Approximation Methods
```@docs
ACA
AdaptiveCrossApproximation.ACAᵀ
iACA
```
#### Pivoting Strategies
```@docs
MaximumValue
AdaptiveCrossApproximation.MaximumValueFunctor
AdaptiveCrossApproximation.RandomSamplingPivoting
AdaptiveCrossApproximation.RandomSamplingPivotingFunctor
FillDistance
AdaptiveCrossApproximation.FillDistanceFunctor
Leja2
AdaptiveCrossApproximation.Leja2Functor
MimicryPivoting
AdaptiveCrossApproximation.MimicryPivotingFunctor
TreeMimicryPivoting
AdaptiveCrossApproximation.TreeMimicryPivotingFunctor
AdaptiveCrossApproximation.CombinedPivStrat

```
#### Convergence Criteria
```@docs
FNormEstimator
AdaptiveCrossApproximation.FNormEstimatorFunctor
iFNormEstimator
AdaptiveCrossApproximation.iFNormEstimatorFunctor
AdaptiveCrossApproximation.FNormExtrapolator
AdaptiveCrossApproximation.FNormExtrapolatorFunctor
AdaptiveCrossApproximation.RandomSampling
AdaptiveCrossApproximation.RandomSamplingFunctor
AdaptiveCrossApproximation.CombinedConvCrit
```

## Functions
```@docs
AdaptiveCrossApproximation.aca
AdaptiveCrossApproximation.acaᵀ
AdaptiveCrossApproximation.nextrc!
```

### Pivoting Functions

```@docs
AdaptiveCrossApproximation.leja2!
AdaptiveCrossApproximation.normF!
AdaptiveCrossApproximation.findcluster
AdaptiveCrossApproximation.center

AdaptiveCrossApproximation.tolerance
```