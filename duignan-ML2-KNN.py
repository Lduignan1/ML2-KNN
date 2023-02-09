#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import sys
import argparse
import math
from math import *
from collections import defaultdict, Counter


# Because of calculation errors, we sometimes end up with negative distances.
# We take here a minimal value of distance, positive (to be able to take the root) and not null (to be able to take the inverse).
MINDIST =  1e-18


class Example:
    """
    An example : 
    vector = vector representation of an object (Ovector)
    gold_class = gold class for this object
    """
    def __init__(self, example_number, gold_class):
        self.gold_class = gold_class
        self.example_number = example_number
        self.vector = Ovector()

    def add_feat(self, featname, val):
        self.vector.add_feat(featname, val)
        #self.vector.get_norm_square()


class Ovector:
    """
    Vector representation of an object to classify
    members
    - f= simple dictionnary from feature names to values
         Absent keys correspond to null values
    - norm_square : square value of the norm of this vector
    """
    def __init__(self):
        self.f = {}
        self.norm_square = 0 

    #def get_norm_square(self):
     #   self.norm_square = sum([val**2 for feat, val in self.f.items()])

    def add_feat(self, featname, val=0.0):
        self.f[featname] = val


    def prettyprint(self):
        # sort features by decreasing values (-self.f[x])
        #           and by alphabetic order in case of equality
        for feat in sorted(self.f, key=lambda x: (-self.f[x], x)):
            print(feat+"\t"+str(self.f[feat]))

    def distance_to_vector(self, other_vector):
        """ Euclidian distance between self and other_vector, 
        Requires: that the .norm_square values be already computed """
        # compute squared of norm of vector; norm_square
        # NB: use the calculation trick
        #   sigma [ (ai - bi)^2 ] = sigma (ai^2) + sigma (bi^2) -2 sigma (ai*bi) 
        #                          = norm_square(A) + norm_square(B) - 2 A.B
        return math.sqrt(self.norm_square + other_vector.norm_square \
             - (2 * (self.dot_product(other_vector))))
        

    def dot_product(self, other_vector):
        """ Returns dot product between self and other_vector """
        # take intersection of keys 
        res = 0
        for feat, val in self.f.items():
            if (feat in other_vector.f):
                res += val * other_vector.f[feat]
        return res

    def cosine(self, other_vector):
        """ Returns cosine of self and other_vector """
        return 1 - self.dot_product(other_vector) / (math.sqrt(self.norm_square) \
             * math.sqrt(other_vector.norm_square))


class KNN:
    """
    K-NN for document classification (multiclass classification)

    members = 

    K = the number of neighbors to consider for taking the majority vote

    examples = list of Example instances

    """
    def __init__(self, examples, K=1, weight_neighbors=None, use_cosine=False, trace=False):
        # examples = list of Example instances
        self.examples = examples
        # the number of neighbors to consider for taking the majority vote
        self.K = K
        # boolean : whether to weight neighbors (by inverse of distance) or not
        self.weight_neighbors = weight_neighbors

        # boolean : whether to use cosine similarity instead of euclidian distance
        self.use_cosine = use_cosine

        # whether to print some traces or not
        self.trace = trace
        

    def classify(self, ovector):
        """
        K-NN prediction for this ovector,
        for k values from 1 to self.K

        Returns: a K-long list of predicted classes, predicted class using K-nearest
        neighbors
        the class at position i is the K-NN prediction when using K=i
        """
        all_distances_ovector = list() # store tuples containing (dist, gold)
        for ex in self.examples:
            # compute either cosine or euclidean distance
            if self.use_cosine:
                all_distances_ovector.append((ovector.cosine(ex.vector), ex.gold_class))
            else:
                all_distances_ovector.append((ovector.distance_to_vector(ex.vector), ex.gold_class))

        all_distances_ovector.sort() # sort list in ascending order 
        counts = defaultdict(int) # dict(class: count)
        k_predicted_classes = list()

        for k in range(self.K):
            if self.weight_neighbors:
                # get freq of each class of first k values of all_distances_ovector
                first_k_distances = all_distances_ovector[:k+1]
                #print(first_k_distances)
                class_frequencies = defaultdict(int) # dict(class: sum of inverse dist)
                for d in first_k_distances:
                    # get sum of inverse distances for each k nearest class 
                    class_frequencies[d[1]] += 1 / d[0]
                #print(class_frequencies)

                k_predicted_classes.append(max(class_frequencies, key=class_frequencies.get))
            else:
                counts[all_distances_ovector[k][1]] += 1 
                # get list of all classes with same max count
                max_ties = sorted([class_ for class_, val in counts.items() if val == max(counts.values())])
                # as list is sorted alphabetically, choose first element to append
                k_predicted_classes.append(max_ties[0])

        return k_predicted_classes, k_predicted_classes[self.K - 1]


    def evaluate_on_test_set(self, test_examples):
        """ Runs the K-NN classifier on a list of Example instances
        and evaluates the obtained accuracy

        Returns: a K-long list of accuracies,
        the accuracy at position i is the one obtained when using K=i
        """
        eval_list = []

        k_corr_counts = defaultdict(int) # keys: k_value, values: number of correct predictions
        for ex in test_examples:
            k_predictions = self.classify(ex.vector)[0] # get k-long list of predicted classes for a given example 
            for k in range(self.K): 
                if k_predictions[k] == ex.gold_class: # check if class at k-index matches gold class of example
                    k_corr_counts[k] += 1

        for k in range(self.K):
            # compute accuracies over each key (k_value) of k_corr_counts
            acc = (f"{(k_corr_counts[k]/len(test_examples)) * 100}% ({k_corr_counts[k]}/{len(test_examples)})") 
            eval_list.append(acc)

            print(f"ACCURACY FOR k = {k+1} = {acc}")

        return eval_list



class Indices:
    """- mapping each word to word integer identifiers from 0 to d-1
    and mapping class to class id"""
    def __init__(self):
        w2i = {} # dict storing word_string: id
        i2w = [] # at rank k in list, storing word with id=k

    def update_w2i(self):
        pass

    def update_i2w(self):
        pass    
        
def read_examples(infile):
    """ Reads a .examples file and returns a list of Example instances """
    stream = open(infile)
    examples = []
    example = None
    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]
        if line.startswith("EXAMPLE_NB"):
            if example != None:
                examples.append(example)
            cols = line.split('\t')
            gold_class = cols[3]
            example_number = cols[1]
            example = Example(example_number, gold_class)
        elif line and example != None:
            (featname, val) = line.split('\t')
            example.add_feat(featname, float(val))
            
    if example != None:
        examples.append(example)
        #example.vector.get_norm_square()

    vocab_indices = Indices()
    for ex in examples:
        for feat, val in ex.vector.f.items():
            ex.vector.norm_square += val**2


    return examples



usage = """ K-NN DOCUMENT CLASSIFIER

  """+sys.argv[0]+""" [options] TRAIN_FILE TEST_FILE

  TRAIN_FILE and TEST_FILE are in *.examples format

"""

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('train_file', default=None, help='Examples that will be used for the K-NN prediction (in .examples format)')
parser.add_argument('test_file', default=None, help='Test examples de test (in .examples format)')
parser.add_argument('-k', "--k", default=1, type=int, help='Hyperparameter K : maximum number of neighbors to consider (all values between 1 and k will be tested). Default=1')
parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
parser.add_argument('-w', "--weight_neighbors", action="store_true", help="If set, neighbors will be weighted before majority vote. If cosine: cosine weighting, if distance, weighting using the inverse of the distance. Default=False")
parser.add_argument('-c', "--use_cosine", action="store_true", help="Toggles the use of cosine similarity instead of euclidian distance. Default=False")
args = parser.parse_args()

#------------------------------------------------------------
# Loading training examples
training_examples = read_examples(args.train_file)
# Loading test examples
test_examples = read_examples(args.test_file)

myclassifier = KNN(examples = training_examples,
                   K = args.k,
                   weight_neighbors = args.weight_neighbors,
                   use_cosine = args.use_cosine,
                   trace=args.trace)

# classification and evaluation on test examples
myclassifier.evaluate_on_test_set(test_examples)
#print(accuracies)
ex_1 = training_examples[0]
ex_1_vec = ex_1.vector
ex_2 = training_examples[1]
ex_2_vec = ex_2.vector


#print(myclassifier.classify(ex_1_vec))
#print((ex_1_vec.norm_square + ex_2_vec.norm_square) - (2 * ex_1_vec.dot_product(ex_2_vec)))
#print(ex_1_vec.dot_product(ex_2_vec))
#print(ex_1_vec.distance_to_vector(ex_2_vec))
