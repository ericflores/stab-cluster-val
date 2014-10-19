stab-cluster-val
================

This module implements a modular version of stability-based cluster validation using Cluster Validation by Prediction Strength presented by Tibshirani, R., & Walther, G. (2005). 

All the four building blocks used by stability-based cluster validation algorithms will be available to be overriden in a child class to allow future researchers to easily implement variations of any of the existing or future stability-based clustering validation methods.

The four building blocks are: 
1. Generate Variants 
2. Assign Labels 
3. Measure Label Concurrency 
4. Measure (In)Stability.
