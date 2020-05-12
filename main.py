import numpy as np
import matplotlib.pyplot as plt

# N : number of data
# Nc : requested number of cluster
# epsilon : convergence parameter, very weak, less than 10^-4
# T : tradeoff parameter // tradeoff between energy and entropy like terms


def init_prob(nbData, nbCluster):
    t = np.random.sample( (nbData, nbCluster) )
    return np.apply_along_axis(lambda a : np.divide(a, np.sum(a)), axis=1, arr=t)

s = np.genfromtxt("data/DS1/simil_ds1.d")
"""
print(np.shape(s))

print(np.shape(s[:, 0]))
print(np.shape(s[:, 1]))
print()
print(s[:10,0])
print()
print(s[:10, 1])
"""


s = np.loadtxt("data/DS1/scatter_ds1.d")
print(np.shape(s))



def gen_cluster(similarity_matrix, T, nbCluster, epsilon):
    nbData = np.shape(similarity_matrix)[0]
    proba_cluster = init_prob(nbData, nbCluster)
    
    while True:
        for i in range(nbData):
            for N in range(nbCluster):
                



#show result data
"""
s = np.loadtxt("data/DS1/scatter_ds1.d")

mapColor = {
    0:'r',
    1:'b',
    2:'g'
}

labelColor = [mapColor[l] for l in s[:,2]]

plt.scatter(s[:,0],s[:,1], c=labelColor)
plt.show()
"""