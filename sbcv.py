# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 13:47:50 2014

@author: Eric Flores

In this investigation we will implement a modular version
of stability-based cluster validation using Cluster Validation
by Prediction Strength presented by Tibshirani (2005), and
evaluate how it performs by identifying correct clusters in
varying types of real datasets. All the four building blocks
used by stability-based cluster validation algorithms will be
available to be overriden in a child class to allow future
researchers to easily implement variations of any of the
existing or future stability-based clustering validation
methods. The four building blocks are:
1. Generate Variants
2. Assign Labels
3. Measure Label Concurrency
4. Measure (In)Stability
"""

#THE FOLLOWING IMPORTS ARE REQUIRED BY THE PARENT CLASS
from __future__ import division
import sys
#from pprint import pprint
import numpy as np

#THESE IMPORTS ARE REQUIRED BY THE PS CHILD CLASS
#Required by StabilityValidationPS.generate_variants
from sklearn.cross_validation import train_test_split
#Required by StabilityValidationPS.assign_labels
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
#Required by StabilityValidationPS.measure_label_concurrency
from sklearn.metrics import pairwise_distances_argmin


#Define all my helper functions here
def main():
    """Main entry point for the script. This module is not designed
    to be called directly. Please use the existing classes and
    methods instead."""



#THIS IS THE PARENT CLASS - DO NOT MODIFY!!!
class StabilityValidation(object):
    """This class defines the parent framework required to perform the
    stability-based cluster validation routine. It provides utility
    functions as well as the method the call the four (4) key methods.
    It includes boilerplate code for those 4 methods, but the actual
    implementation of these methods must be specified in the child class.
    Usage:

    find_cluster_num(data, n_runs = 10, max_clusters=10, clustering_method='KMeans', **other)

    Where:
        data - Is the Numpy array (or Pandas Dataframe) having the data
        to be clustered.
        n_runs - Specifies the number of cross-validations to be performed.
        max_clusters - The maximum number of clusters to attemp.
        clustering_method - A string with the name of the Scikit-Learn
        clustering altorithm to be called.
        other - This means that the function accept one or more additional
        key=value pairs to be passed to the clustering altorithm.

    Example #1: Agglomerative Clustering with Cosine Affinity and Average Linkage

    result = objectname.find_cluster_num(mydata, n_runs = 1000, \
             max_clusters=5, clustering_method='AgglomerativeClustering', \
             linkage='average', affinity='cosine')

        Important: The example above will only work if the objectname is
        an object derived from the child class, as follows:
        objectname=sbcv.StabilityValidationPS()
        If a different child class is defined, then it can be used too.
    """

    def generate_variants(self, singlearray):
        """This function is a template to be overridden in a child class.
        It is supposed to take a NumPy array as input, split the records
        randomly and return a list having as many arrays as variants
        where defined."""
        listofarrays = []
        return listofarrays

    def assign_labels(self, listofarrays):
        """This function is a template to be overridden in a child class.
        It takes a list with as many arrays as were created in the
        generate_variants function and will return a list with as many
        label vectors as arrays existed in the input list.
        Actually, the return are 1-dimension arrays, to be exact."""
        listofvectors = []
        return listofvectors

    def measure_label_concurrency(self, listofarrays, assignations):
        """This function is a template to be overridden in a child class.
        It takes a list with many arrays and another list of many vectors
        (1-d arrays). Both list must be exactly the same length.
        This function return a scalar with the resulting
        metric measuring how similar are the cluster found among the
        two or more cluster runs."""

    def measure_in_stability(self, dictwithruns):
         """This function is a template to be overridden in a child class.
         It takes a dictionary having keys that identify the specific run.
         For example "k=2", "k=3", etc. The dictionary values are lists
         of the label concurrency metrics obtained. If the trial for k=2
         is performed 10 times, then this list will have 10 scalars."""

    def find_cluster_num(self, data, n_runs = 10, max_clusters=10, clustering_method='KMeans', **other):
        """This is the main function implementing the Stability Based Clustering Validation search
        for optimal number of clusters. It returns a dictionary having with results calculated for 
        each value of number of clusters."""
        results = {}
        for n_clusters in range(2, max_clusters + 1):
            #trials = []
            trials = {'runs':[]}
            lastitem=-1
            for run in range(n_runs):
                trials['runs'].append(self.generate_variants(data))
                trials['runs'][lastitem]=self.assign_labels(trials['runs'][lastitem], method=clustering_method, k=n_clusters, **other)
                trials['runs'][lastitem]=self.measure_label_concurrency(trials['runs'][lastitem])
            trials = self.measure_in_stability(trials)
            results[n_clusters]=trials
        return results

    def find_cluster_center(self, points):
        return points.mean(axis=0)


class StabilityValidationPS(StabilityValidation):
    """This class inherits most methods from the StabilityValidation
    class and overrides key methods to perform the Tibshirani's
    Prediction Strength. This is a reference implementation to be
    used as example for implementing other future stability-based
    cluster validation methods."""

    def generate_variants(self,singlearray):
        """This function implements at 50/50 split using a  function
        from Scikit Learn's Cross Validation module. It return a list
        with two arrays."""
        return {'variants':train_test_split(singlearray, test_size=0.5)}

    def assign_labels(self, dict_with_variants, method='KMeans', k=2, **other):
        """This function will take a dict with an element called
        'variants'. This element consist of a list with two arrays
        The method will perform the same clustering method to both.
        It also requires the name of the sklearn function that will
        perform the clustering (make sure to import it first) and
        the value of k. Optionally, pass a one or more key=values
        with other parameters to be passed to the clustering function.
        Currently this function only supports the following methods:
        1. 'KMeans'
        2. 'SpectralClustering'
        3. 'AgglomerativeClustering'
        4. 'MiniBatchKMeans' (actually...not tested, but should work)
        The method will return a dictionary with the following elements:
        variants: The same list received as first parameter in the input.
        labels: A list with two 1-d arrays having the labels of the two sets
        centers: A list of lists. The first level is one element per
                 dataset variant and the second level has one element per
                 cluster.
        """
        variantlist = dict_with_variants.get('variants')
        if variantlist <> None:
            #Do we really have an optional parameter? It is only one string?
            optional_parameter = ''
            if len(other)> 0:                         #Do we have extra arguments?
                optional_parameter=',' + ', '.join("%s=%r" % (key,val) for (key,val) in other.iteritems())
            ks=str(k)
            #Prepare the clustering object and set the runtime parameters
            clustering_string=method + "(n_clusters=" + ks + optional_parameter + ")"
            clusteringmodeler = eval(clustering_string)
            clusterlist = []
            vectorlist = []
            centerlist = []
            for variant_no in range(len(variantlist)):
                clusterlist.append(clusteringmodeler.fit(dict_with_variants['variants'][variant_no]))
                vectorlist.append(clusterlist[variant_no].labels_)
                centerlist.append(np.array([self.find_cluster_center(variantlist[variant_no][vectorlist[variant_no]==mycluster]) for mycluster in set(vectorlist[variant_no])]))
        return dict(dict_with_variants.items() + {'labels':vectorlist, 'centers':centerlist}.items())

    def measure_label_concurrency(self, assignations):
        """It takes a dictionary with 'variants', 'labels' and 'centers'
        This function return the same dictionary after adding the
        metric measuring how similar are the cluster found among the
        two or more cluster runs.
        This function implements the pairwise concurrency metric
        described in Tibshirani's Prediction Strength."""
        test_points=assignations['variants'][1]
        test_labels=assignations['labels'][1]
        train_centers=assignations['centers'][0]
        clustermetric=[]
        #This loop goes over each cluster in the test set
        for clusternum in set(test_labels):
            #Prepare subset having only the data points for current cluster
            clusterset=test_points[test_labels==clusternum]
            clustersize=len(clusterset)
            if clustersize>1:
                #The line below finds the nearest Train set cluster center
                #for each data point of this Test cluster
                membership=pairwise_distances_argmin(clusterset, train_centers)
                #Placeholder for comembership cummulative value
                comembership=0
                #These two loops will compare cluster center of each data point
                #in the cluster to all other data points in the same cluster
                for i in range(len(membership)):
                    for j in range(len(membership)):
                        #If testing two different data points and they share nearest Train cluster centr
                        if i<>j and membership[i]==membership[j]:
                            #Then incremement metric
                            comembership=comembership+1
                clustermetric.append(comembership/(clustersize*(clustersize-1)))
            pass
        return dict(assignations.items() + [('metric', min(clustermetric))])


    def measure_in_stability(self, dictwithruns):
        """It takes a dictionary having 'runs' element that contain
        the list of dictionarieskeys that identify the specific run.
        For example "k=2", "k=3", etc. The dictionary values are lists
        of the label concurrency metrics obtained. If the trial for k=2
        is performed 10 times, then this list will have 10 scalars."""
        listofmetrics = np.array([ run['metric'] for run in dictwithruns['runs']])
        return_dict = dictwithruns
        return_dict['metric_center']=listofmetrics.mean()
        return_dict['metric_spread']=listofmetrics.std()
        return return_dict

    def test_measure_label_concurrency(self):
        """This is an utility function used for testing the method
        that implements the pairwise concurrency method in the
        child class (measure_label_concurrency).

        It defines a baseline test scenario where it is expected to
        return a Prediction Strength of 1, and modifies it progressively
        to obtain the Prediction Strength of the degraded examples.
        These results can be compared with the expected values."""
        test_scenario={
        #We define below two identical datasets. They each represent
        #a training and test set.
        'variants': [
        np.array([[0,0],[0,0],[0,0],[0,0],[1,1],[1,1],[1,1],[1,1]]),
        np.array([[0,0],[0,0],[0,0],[0,0],[1,1],[1,1],[1,1],[1,1]]) ]
        ,
        #The vectors below represent the cluster assignment for each
        #point in the training and test sets, respectively.
        'labels': [
        np.array([0,0,0,0,1,1,1,1]),
        np.array([0,0,0,0,1,1,1,1]) ]
        ,
        #Below are the cluster centers for each of the datasets
        'centers': [
        np.array([[0,0],[1,1]]),
        np.array([[0,0],[1,1]]) ]
        }
        test_results = []
        #Get the metric for when all data points in test set match
        #centers in the training set
        test_results.append(self.measure_label_concurrency(test_scenario)['metric'])
        #Change a point in the test set so that it does not match
        #the center in the training set; then capture the value.
        #The expected metric is 0.5.
        test_scenario['variants'][1][0,:]=np.array([1,1])
        test_results.append(self.measure_label_concurrency(test_scenario)['metric'])
        #Change a second point. The expected metric is 0.333...
        test_scenario['variants'][1][1,:]=np.array([1,1])
        test_results.append(self.measure_label_concurrency(test_scenario)['metric'])
        #Change a third point. The metric should be 0.5 again.
        test_scenario['variants'][1][2,:]=np.array([1,1])
        test_results.append(self.measure_label_concurrency(test_scenario)['metric'])
        print "Expected Results:\n[1.0, 0.5, 0.3333333333333333, 0.5]"
        print "Actual Results:\n", test_results

#Here starts the actual execution of them script if it ever is called directly
if __name__ == '__main__':
    sys.exit(main())


