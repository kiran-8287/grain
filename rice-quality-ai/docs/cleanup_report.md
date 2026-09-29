# Cleanup Report

## Removed

- Datasets and dataset archives
- Annotation and mask assets
- Training code and data-prep scripts
- Model checkpoint and weight directories
- Evaluation and experiment outputs
- Obsolete ML tests
- Dataset/training documentation
- Legacy model and dataset dependencies

## Preserved

- Backend API and service layer
- Frontend application shell
- Rice analysis utilities and quality rules
- Core configuration files required by the application
- General project documentation not tied to the retired training pipeline

## Verification

- Test result: pending final repository validation
- Build result: pending final repository validation
- Lint/type-check result: pending final repository validation
- Remaining ML files: none specific to the old training pipeline
- Remaining dataset files: none committed in the repository
- Remaining model files: none committed in the repository

The repository is left as a clean baseline for a fresh dataset selection and training implementation.
