import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Build the send feature set and the associated data and labels
pair1 = (auto_data,auto_labels)

features2 = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

auto_data2, auto_labels2 = hw3.auto_data_and_labels(auto_data_all, features2)
pair2 = (auto_data2,auto_labels2)

# Run over the learner algorithm
arr = np.zeros((2,2))
for learn_ind, learner in enumerate([hw3.perceptron,hw3.averaged_perceptron]):  
    for pair_ind, pair in enumerate([pair1,pair2]):
        arr[learn_ind,pair_ind]= hw3.xval_learning_alg(learner, pair[0], pair[1], 10)

## Printing the result, a row corresponds to a (perceptron, avraged_perceptron) pair
#print('Result table: ', arr.T)

# Evaluating the classifier for the best case (T=10,feature2)
th, th0 = hw3.averaged_perceptron(auto_data2, auto_labels2, params = {'T':10}, hook = None)

## Printing result (each pair of indecies correspond to a single feature)
#print('(theta,theta0: ',(th,th0))
#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Solution to the first part
# arr = np.zeros((2))
# for learn_ind, learner in enumerate([hw3.perceptron,hw3.averaged_perceptron]):  
#         arr[learn_ind]= hw3.xval_learning_alg(learner, review_bow_data, review_labels, 10)

## Printing the results of the first part
#print('Review table:', arr) 


## Solution to the second part
th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params = {'T':10}, hook = None)


#th.argsort()[-10:]
key_words_ind = np.argsort(th[:,0])
pos_words_ind = key_words_ind[-10:]
neg_words_ind = key_words_ind[:10]

word_dict = hw3.reverse_dict(dictionary)
pos_words = [word_dict[ind] for ind in pos_words_ind]
neg_words = [word_dict[ind] for ind in neg_words_ind]

#print('most positive words: ',pos_words)
#print('most negative words: ',neg_words)


#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples = x.shape[0]
    m = x.shape[1]
    n = x.shape[2]
    return  x.reshape(n_samples,m*n).T

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    n_samples, m, n = x.shape
    return np.mean(x,axis = 2,keepdims=True).T.reshape((m,n_samples))


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    n_samples, m, n = x.shape
    return np.mean(x,axis = 1,keepdims=True).T.reshape((n,n_samples))


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    n_samples, m, n = x.shape  
    top = np.mean(x[:,:m//2,:],axis = (1,2),keepdims=True).T.reshape((1,n_samples))
    bottom = np.mean(x[:,m//2:,:],axis = (1,2),keepdims=True).T.reshape((1,n_samples))
    return np.concatenate((top,bottom))
                          

                      
# print(a.shape)
# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)


#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------
print('----------------')
print('raw accuracy: ', acc,'\n')

acc_row = hw3.get_classification_accuracy(row_average_features(data), labels)
acc_col = hw3.get_classification_accuracy(col_average_features(data), labels)
acc_top = hw3.get_classification_accuracy(top_bottom_features(data), labels)
# print('row accuracy: ', acc_row,'\n')
# print('column accuracy: ', acc_col,'\n')
# print('top/bottom accuracy: ', acc_top,'\n')
print([acc_row,acc_col,acc_top])

