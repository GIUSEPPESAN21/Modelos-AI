# Conditional GAN (cGAN) — Counterfactual Health Profiles

## Architecture

```
Generator: (noise_z, condition) → synthetic feature vector (20-dim)
Discriminator: (features, condition) → real/fake
Condition: normalized year, GDP, urbanization
```

## Use Cases

- Generate plausible counterfactual health profiles under policy scenarios
- Augment sparse regions of feature space for downstream models

## Limitations

- Mode collapse risk with small data
- No formal causal identification (pair with DML)
- Generated values may violate physical constraints
