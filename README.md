**Hierarchical VAE–Symbolic Regression for Multistage Rubber Processing**

  This repository provides the Python implementation of an interpretable modeling framework integrating hierarchical variational autoencoders (VAE) and symbolic regression (SR) to investigate process–response–property relationships in multistage rubber processing.
  The framework combines stage-wise feature selection, hierarchical latent representation learning, sample-adaptive gating, and explicit symbolic regression to characterize relationships among processing conditions, mixing responses, curing responses, and material properties, with Shore A hardness as the primary prediction target.

**1. Curing_curve-1.csv**
  Raw experimental curing-curve data for Sample 1, containing time-dependent measurements recorded during the curing test.
  
**2. Mixing_curve-1.csv**
  Raw experimental mixing-curve data for Sample 1, containing time-dependent measurements recorded during the mixing process.
  
**3. data.csv**
  An example processed data record containing extracted experimental features, demonstrating the structured input format used for hierarchical representation learning and property prediction.
  
**4. hardness_ha_interpretability.txt**
  A modeling and interpretability report for Shore A hardness, including selected features, cross-validation results, hierarchical gating statistics, latent-state associations, symbolic regression equations, feature-selection stability, and model diagnostics.
  
**5. main-HierarchicalVAE_SR_Hardness.py**
  The main Python implementation of the hierarchical VAE–SR framework, covering stage-wise feature processing and selection, hierarchical representation learning, hardness prediction, symbolic regression, and model evaluation.
  
**6. raw.csv:**
  An example raw dataset containing two experimental records with specified mixing and curing conditions, illustrating the original processing parameters prior to feature extraction.
Data Availability Note

  The example datasets are provided to illustrate the organization, feature extraction, and processing of multistage rubber experimental data. They represent only a subset of the experimental information and do not constitute the complete dataset used in the study. Consequently, the example files alone are not sufficient to reproduce all reported modeling results.

**Computational Environment**
  The hierarchical VAE–SR framework was implemented in Python 3.10.20 and executed in a Jupyter Notebook environment. All model training, prediction, and evaluation procedures were performed on a CPU without GPU acceleration. The implementation uses PyTorch for hierarchical representation learning and integrates machine learning and symbolic regression methods for feature selection, hardness prediction, and model interpretation.
