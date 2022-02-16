---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Notebook showing how to use Azure Machine Learning pipelines to do Batch Predictions with an Image Classification model trained using AutoML.
---

# Batch Scoring with an Image Classification Model
- Dataset: Toy dataset with images of products found in a fridge
    - **[Jupyter Notebook](auto-ml-image-classification-multiclass-batch-scoring.ipynb)**
        - register an Image Classification Multi-Class model already trained using AutoML
        - create an Inference Dataset
        - provision compute targets and create a Batch Scoring script
        - use ParallelRunStep to do batch scoring
        - build, run, and publish a pipeline
        - enable a REST endpoint for the pipeline
