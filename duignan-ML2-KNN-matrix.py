#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# c'est moche mais ça marche
import sys
import argparse
import math
import numpy as np
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

    X_train = T x V matrix whose i-th row is the vector for the i-th example
    Y_train = vector of size T, for gold classes of T examples

    """
    def __init__(self, X_train, Y_train, K=1, weight_neighbors=None, use_cosine=False, trace=False):
        # examples = list of Example instances
        #self.examples = examples

        self.X_train = X_train

        self.Y_train = Y_train

        # the number of neighbors to consider for taking the majority vote
        self.K = K
        # boolean : whether to weight neighbors (by inverse of distance) or not
        self.weight_neighbors = weight_neighbors

        # boolean : whether to use cosine similarity instead of euclidian distance
        self.use_cosine = use_cosine

        # whether to print some traces or not
        self.trace = trace
        
    def dist(self, X_test):
      """return euclidean distance between all test and all train vectors"""

      norm_squares_test = np.sum(X_test**2, axis=1, keepdims=True)
      norm_squares_train = np.sum(self.X_train**2, axis=1, keepdims=True)

      return np.sqrt(norm_squares_test + norm_squares_train.transpose() -2 * X_test.dot(X_train.transpose()))

    def cosine(self, X_test):
      """return matrix containing dot products of normalized row vectors in X_test and rows in X_train"""
      X_train_normalized = np.apply_along_axis(normalize, 1, self.X_train)
      X_test_normalized = np.apply_along_axis(normalize, 1, X_test)

      return 1 - np.dot(X_test_normalized, X_train_normalized.transpose())


    def classify(self, X_test):

        """
        K-NN prediction for this ovector,
        for k values from 1 to self.K

        Returns: a K-long list of predicted classes, predicted class using K-nearest
        neighbors for each example in the test set
        the class at position i is the K-NN prediction when using K=i
        """


        all_distances_matrix = list() # store tuples containing (dist, gold)
  
        if self.use_cosine:
            dist_matrix = self.cosine(X_test)
            #all_distances_ovector.append((ovector.cosine(ex.vector), ex.gold_class))
        else:
            dist_matrix = self.dist(X_test)
            #all_distances_ovector.append((ovector.distance_to_vector(ex.vector), ex.gold_class))

        
        for row in dist_matrix:
          dist_gold_row = list() # store dist, gold in a list for each doc
          for dist, gold in zip(row, self.Y_train):
            dist_gold_row.append((dist, gold))

          dist_gold_row.sort() # sort row in ascending order
          all_distances_matrix.append(dist_gold_row)


        k_pred_matrix = list()

        for dist_gold_row in all_distances_matrix:
          counts = defaultdict(int) # dict(class: count)
          k_predicted_classes = list()

          for k in range(self.K):

            counts[dist_gold_row[k][1]] += 1 

            # get list of all classes with same max count
            max_ties = sorted([class_ for class_, val in counts.items() if val == max(counts.values())])
            # as list is sorted alphabetically, choose first element to append
            k_predicted_classes.append(max_ties[0])
            #print(k_predicted_classes)
          
          k_pred_matrix.append(k_predicted_classes)

          

        return k_pred_matrix



    def evaluate_on_test_set(self, X_test, Y_test):
        """ Runs the K-NN classifier on a list of Example instances
        and evaluates the obtained accuracy

        Returns: a K-long list of accuracies,
        the accuracy at position i is the one obtained when using K=i
        """
        eval_list = []

        k_corr_counts = defaultdict(int) # keys: k_value, values: number of correct predictions
        
        k_pred_matrix = self.classify(X_test)
        for pred_list, gold_class in zip(k_pred_matrix, Y_test):
          for k in range(self.K):
            if pred_list[k] == gold_class:
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
        self.w2i = {} # dict storing word_string: id
        self.i2w = [] # at rank k in list, storing word with id=k

    def update_w2i(self, word, index):
        self.w2i[word] = index

    def update_i2w(self, word):
        self.i2w.append(word)
        
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
            ex.vector.norm_square += val**2 # computing vector norm squares

            # mapping words to ids and vice-versa
            if feat not in vocab_indices.i2w:
                vocab_indices.update_i2w(feat)
                vocab_indices.update_w2i(feat, vocab_indices.i2w.index(feat))


    #print(vocab_indices.i2w)
    
    return examples, vocab_indices

def get_X_and_Y(examples, indices):
    """input: list of Examples instances + previously built Indices instance
    output: corresponding X and Y structures"""


    X = np.zeros((len(examples), len(indices.i2w)))
    Y = [0] * len(examples)

    for ex_ind, ex in enumerate(examples):
        Y[ex_ind] = ex.gold_class

        for word, word_ind in indices.w2i.items():
            if word in ex.vector.f:
                X[ex_ind, word_ind] = ex.vector.f[word]

    Y = np.array(Y)
    return X, Y

def normalize(vector):
    """return a normalized version of a vector represented as 1d numpy array"""
    return vector / np.linalg.norm(vector)


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
# Loading training examples and indices
training_examples, training_indices = read_examples(args.train_file)
# Loading test examples
test_examples, test_indices = read_examples(args.test_file)

X_train, Y_train = get_X_and_Y(training_examples, training_indices)
X_test, Y_test = get_X_and_Y(test_examples, training_indices) # reuse training indices to ignore new words

myclassifier = KNN(X_train=X_train, 
                   Y_train=Y_train,
                   K = 5,
                   weight_neighbors = False,
                   use_cosine = True,
                   trace=False)

# classification and evaluation on test examples
myclassifier.evaluate_on_test_set(X_test, Y_test)
#print(myclassifier.classify(X_test))