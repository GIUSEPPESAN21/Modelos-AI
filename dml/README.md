# Double/Debiased Machine Learning (DML)

## Method

Partially linear regression with cross-fitted nuisance models:
- **g(X)**: outcome model E[Y|X]
- **m(X)**: treatment model E[D|X]
- **ATE θ**: coefficient from regressing (Y - ĝ) on (D - m̂)

## Libraries

- **EconML**: `LinearDML` with random forest nuisances
- **DoubleML**: `DoubleMLPLR` with Lasso nuisances
- Manual cross-fitted implementation as fallback

## Treatment

Synthetic `health_policy_treatment`: high health expenditure (>6.5% GDP) post-2010.

## Limitations

- Synthetic treatment; not real policy variation
- Unconfoundedness assumption not testable
- Small sample limits nuisance model complexity
