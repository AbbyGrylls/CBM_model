"""
FEATURE SCALING STRATEGY REPORT
===============================

This document describes the feature-wise scaling strategy adopted for the dataset,
based on exploratory data analysis (EDA) using histogram distributions.

Scaling decisions are NOT model-agnostic defaults; instead, they are driven by:
- Distribution shape
- Presence of skewness
- Outliers
- Physical bounds
- Feature semantics

Each feature group below includes:
1. Observations from histograms
2. Identified statistical issues
3. Justification for chosen transformation and scaler
"""

# -------------------------------------------------------------------
# 1. PRESSURE-RELATED FEATURES
# -------------------------------------------------------------------
"""
Features:
- Pressure (kPa)
- Max. EXP Pressure (kPa)
- Langmuir Pressure Constant, PL (kPa)

Histogram Observations:
- Strong right-skewed distributions
- Majority of samples clustered at low pressure values
- Very long tails extending to high pressure ranges
- Presence of extreme outliers spanning several orders of magnitude

Statistical Issues Identified:
- High variance dominated by large values
- Non-Gaussian behavior
- Outliers significantly influence mean and standard deviation

Scaling Decision:
- Apply logarithmic transformation (log1p)
- Follow with RobustScaler

Justification:
- Log transform compresses extreme values and stabilizes variance
- Converts multiplicative scale effects into additive ones
- RobustScaler uses median and IQR instead of mean and variance,
  making it resistant to remaining outliers after log transformation
- Prevents pressure features from dominating loss gradients
  or distance calculations in ML models
"""

PRESSURE_SCALING = "log1p + RobustScaler"


# -------------------------------------------------------------------
# 2. ADSORPTION VOLUME FEATURES
# -------------------------------------------------------------------
"""
Features:
- EXP_ADS (mL/g)
- Langmuir_ADS (mL/g)
- Langmuir Volume Constant, VL (mL/g)

Histogram Observations:
- Approximately unimodal distributions
- Moderate spread around a central tendency
- Mild skewness but no extreme tails
- Mean and median are relatively close

Statistical Issues Identified:
- No severe skew
- No extreme outliers
- Variance reasonably stable

Scaling Decision:
- Apply StandardScaler

Justification:
- StandardScaler assumes roughly symmetric distributions,
  which is sufficiently satisfied here
- Centers data to zero mean and unit variance
- Preserves relative distances between samples
- Suitable for linear models, SVMs, and neural networks
"""

ADSORPTION_SCALING = "StandardScaler"


# -------------------------------------------------------------------
# 3. PHYSICAL CONTROL VARIABLES
# -------------------------------------------------------------------
"""
Features:
- Temperature (°C)
- Depth (m)

Histogram Observations:
- Temperature shows discrete clustering (experimental setpoints)
- Depth shows a wide but controlled range
- No explosive outliers

Statistical Issues Identified:
- Different physical units and scales
- Bounded by experimental or geological constraints

Scaling Decision:
- Temperature → StandardScaler
- Depth → MinMaxScaler (or StandardScaler depending on model)

Justification:
- Temperature benefits from centering due to discrete bands
- Depth is a monotonic physical variable where relative position
  within range is more important than variance
- MinMaxScaler preserves ordering and physical interpretability
"""

PHYSICAL_SCALING = {
    "Temperature": "StandardScaler",
    "Depth": "MinMaxScaler"
}


# -------------------------------------------------------------------
# 4. PROXIMATE ANALYSIS VARIABLES (PERCENTAGES)
# -------------------------------------------------------------------
"""
Features:
- Moisture (%)
- VM (%)
- Ash (%)
- FC (%)

Histogram Observations:
- Values strictly bounded between 0 and 100
- Some skewness (especially Moisture and Ash)
- No extreme outliers

Statistical Issues Identified:
- Non-Gaussian but bounded distributions
- Physical meaning tied to percentage values

Scaling Decision:
- Apply MinMaxScaler

Justification:
- MinMaxScaler preserves natural bounds of percentage data
- Prevents artificial variance inflation
- Keeps features within [0, 1], which is ideal for neural networks
- Maintains physical interpretability of compositional data
"""

PROXIMATE_SCALING = "MinMaxScaler"


# -------------------------------------------------------------------
# 5. MACERAL COMPOSITION VARIABLES
# -------------------------------------------------------------------
"""
Features:
- Vitrinite (%)
- Semi-vitrinite (%)
- Liptinite (%)
- Inertinite (%)
- Mineral matter (%)

Histogram Observations:
- Strong zero inflation (large spike near zero)
- Highly right-skewed distributions
- Sparse values with occasional large percentages
- Long-tailed behavior

Statistical Issues Identified:
- Non-linearity
- Sparse distributions
- Variance dominated by few samples
- Zero values invalidate simple log transforms

Scaling Decision:
- Apply PowerTransformer (Yeo–Johnson)
- Follow with RobustScaler

Justification:
- Yeo–Johnson transformation:
  - Handles zero and positive values safely
  - Reduces skewness
  - Stabilizes variance
- RobustScaler further protects against remaining extreme values
- Prevents rare maceral-rich samples from biasing the model
"""

MACERAL_SCALING = "PowerTransformer (Yeo–Johnson) + RobustScaler"


# -------------------------------------------------------------------
# 6. CORRELATION METRIC
# -------------------------------------------------------------------
"""
Feature:
- Correlation coefficient (R²)

Histogram Observations:
- Highly concentrated near 1
- Very low variance
- Strictly bounded between 0 and 1

Statistical Issues Identified:
- Near-saturated distribution
- Limited information content

Scaling Decision:
- MinMaxScaler (or no scaling for tree-based models)

Justification:
- Preserves bounded nature
- Keeps numerical compatibility with distance-based models
- Avoids unnecessary distortion of an already normalized metric
"""

CORRELATION_SCALING = "MinMaxScaler"


# -------------------------------------------------------------------
# FINAL REMARKS
# -------------------------------------------------------------------
"""
Key Principle:
--------------
Scaling decisions are driven by distribution characteristics observed
in exploratory analysis, not by convenience or uniform preprocessing.

This feature-specific approach:
- Improves numerical stability
- Reduces bias from outliers
- Preserves physical meaning
- Enhances model generalization

Recommended for:
- Linear models
- SVM
- KNN
- Neural networks

Tree-based models may skip scaling, but transformations
(log / power) can still improve split quality.
"""
