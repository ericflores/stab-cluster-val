stab-cluster-val
================

This module implements a modular version of stability-based cluster validation using Cluster Validation by Prediction Strength presented by Tibshirani, R., & Walther, G. (2005). 

All the four building blocks used by stability-based cluster validation algorithms will be available to be overriden in a child class to allow future researchers to easily implement variations of any of the existing or future stability-based clustering validation methods.

The four building blocks are:

1. Generate Variants
  * This method  takes an object that can be coerced into a Numpy Array and returns a dictionary with an entry labeled **variants**. This content of the **variants** key will be a list with as many variants of the original dataset as required. For the Prediction Strength it just returns a Training and Test subset.
2. Assign Labels
  * This method takes the dictionary returned by Generate Variants and return the same dictionary, having also an entry called **labels**. The content of **labels** is a list with as many list as variants exist. For the Prediction Strength, the two list will have the clustering assignments for the Test and Training dataset. It also returns an optional dictinary entry called **centers**. This is a list of arrays, with each array having the centers for the clusters identified in the Training and Test datasets.
3. Measure Label Concurrency
  * This method takes the dictionary object returned by Assign Labels and calculates the stability based clustering validation metric. For the case of Prediction Strength, it calculates the pairwise cluster concurrency required by this metric and adds it to the input dictionary as a label called **metric**. 
4. Measure (In)Stability
  * This is a summary method. You are supposed to run the #1 to #3 several times, and use this method to summarize the results. Instead of taking the output of Measure Label Concurrency directly, it receives a dictionary with a single entry called **runs**. The content of this entry is a list with the dicts resulting from all the different runs of #1 to #3. This method will return two new entries to the top level dictionary: **metric_center** and **metric_spread**. 
