#Hierarchical VAE–Symbolic Regression for Multistage Rubber Processing

This repository provides the implementation of an interpretable modeling framework integrating hierarchical variational autoencoders (VAE) and symbolic regression (SR) for analyzing process–response–property relationships in multistage rubber processing. The framework combines stage-specific latent representation learning, sample-adaptive gating, and explicit symbolic modeling to investigate associations among processing conditions, mixing responses, curing responses, and final mechanical properties.

Compound-1.csv: Example curing-response curve data for Sample 1, illustrating the experimental signals used for curing-related feature extraction and subsequent modeling.

README.md: Repository documentation providing an overview of the research framework, file organization, experimental data, and model implementation.

Vulcanization-1.csv: Example mixing-response curve data for Sample 1, illustrating the time-dependent experimental measurements used to characterize mixing behavior and extract process-response features.

data.csv: An example processed data record containing extracted experimental features, demonstrating the structured input format used for hierarchical representation learning and property prediction.

hardness_ha_interpretability.txt: A comprehensive modeling and interpretability report for Shore A hardness, including selected features, cross-validation results, hierarchical gating statistics, latent-state associations, explicit symbolic regression equations, feature-selection stability, and model diagnostics.

main-HierarchicalVAE_SR_Hardness.py: The main Python script implementing the hierarchical VAE–SR framework, including multistage feature processing, latent representation learning, hardness prediction, symbolic regression, and model evaluation.

raw.csv: An example raw dataset containing two experimental records of prescribed mixing and curing conditions, illustrating the original processing parameters before feature extraction and model development.

The example datasets are provided to demonstrate the organization and processing of multistage experimental information. They do not represent the complete experimental dataset used in the study.
