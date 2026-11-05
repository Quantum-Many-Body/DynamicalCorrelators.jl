# Observables

## Correlation Plans

```@docs
CorrelationPlan
add_one_site!
add_two_site!
add_pairing!
raw_call_counts
```

## Static Correlators

```@docs
correlator
```

## Dynamical Correlators

```@docs
evolve_mps
dcorrelator
prepare_dcorrelator
sweep_dot
```

## Fourier Transforms

```@docs
fourier_kw
fourier_rw
fourier_rz
fourier_riw
DynamicalCorrelators.fourier_kt
static_structure_factor
```

## Linear Prediction

```@docs
LinearPrediction
LinearPredictionModel
fit_linear_prediction
extend_series
extrapolate_series
validation_error
pole_analysis
```

## Lehmann Representation

```@docs
recommend_tmax
lehmann_eval
lehmann_report
```

## Transport

```@docs
conductivity
```
