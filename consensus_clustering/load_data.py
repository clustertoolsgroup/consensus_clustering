'''
supplies loading functions from example datasets stored in 
./consensus_clustering/data
'''

import numpy as np

def load_fuzzy_data(scale = 'False'):
    '''
    loads dataset from
    	
    https://github.com/scikit-learn-contrib/hdbscan/blob/master/notebooks/clusterable_data.npy
    '''
    if scale == 'True':
        data = scale_data(np.load('./consensus_clustering/data/fuzzy_data.npy'))
    else:
        data =  np.load('./consensus_clustering/data/fuzzy_data.npy')
    return data

def load_aggregation_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/Aggregation.txt'))
    else:
        data = np.loadtxt('./consensus_clustering/data/Aggregation.txt')
    return data

def load_birch1_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/birch1.txt'))
    else:
        data = np.loadtxt('./consensus_clustering/data/birch1.txt')
    return data

def load_birch3_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/birch3.txt'))
    else:
        data = np.loadtxt('./consensus_clustering/data/birch3.txt')
    return data

def load_compound_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/Compound.txt'))
    else:
        data = np.loadtxt('./consensus_clustering/data/Compound.txt')
    return data

def load_flame_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/flame.txt'))
    else:
        data = np.loadtxt('./consensus_clustering/data/flame.txt')
    return data

def load_pathbased_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/pathbased.txt'))
    else:
        data = np.loadtxt('./consensus_clustering/data/pathbased.txt')
    return data

def load_sets_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/s4.txt'))
    else:
        data = np.loadtxt('./consensus_clustering/data/s4.txt')
    return data

def load_spiral_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/spiral.txt'))
    else: 
        data = np.loadtxt('./consensus_clustering/data/spiral.txt')
    return data
    
def load_uneven_blobs(scale = 'False'):
    '''
    uneven blobs dataset generated with sklearn.datasets.make_blobs
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/uneven_blobs.txt'))
    else: 
        data = np.loadtxt('./consensus_clustering/data/uneven_blobs.txt')
    return data

def scale_data(data):
    '''
    scale data to mean 0 and standard deviation 1
    http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
    '''
    [n,d] = np.shape(data)
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    scaled = np.array([(data[:,i]-mean[i])/std[i] for i in range(d)]).T
    return scaled

def load_iris(scale = 'False'):
    '''
    famous (real) iris data set, see e.g. https://en.wikipedia.org/wiki/Iris_flower_data_set
    n=150, d=4 (labels are known, there are three different types of flowers)
    
    downloaded from https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/iris.txt'))
    else: 
        data = np.loadtxt('./consensus_clustering/data/iris.txt')
    return data

def load_iris_labeled(scale = 'False'):
    '''
    famous (real) iris data set, see e.g. https://en.wikipedia.org/wiki/Iris_flower_data_set
    n=150, d=4 (labels are known, there are three different types of flowers)
    
    downloaded from https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./consensus_clustering/data/iris_labels.txt'))
    else: 
        data = np.loadtxt('./consensus_clustering/data/iris_labels.txt')
    return data
