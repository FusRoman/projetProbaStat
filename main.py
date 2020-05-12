import numpy as np
import matplotlib.pyplot as plt

# N : number of data
# Nc : requested number of cluster
# epsilon : convergence parameter, very weak, less than 10^-4
# T : tradeoff parameter // tradeoff between energy and entropy like terms


#génére P(C|i) à m = 0
def init_prob(nbData, nbCluster):
    t = np.random.sample( (nbCluster, nbData) )
    return np.apply_along_axis(lambda a : np.divide(a, np.sum(a)), axis=1, arr=t)

def gen_cluster(similarity_matrix, Nc, T=2, epsilon=0.0001):
    nbData = np.shape(similarity_matrix)[0]
    Pci = init_prob(nbData, Nc)

    while True:
        lastPci = np.copy(Pci)
        
        for i in range(nbData):
            
            #calcul p(i), comme il ont tous la même proba, c'est juste i/N
            Pi = 1/nbData
            
            #calcul P(C) pour tout C
            Pc = np.sum(Pci, axis = 1) * Pi
            
            #calcul s^(m)(C;i) pour tout C
            sCi = np.multiply(np.sum(np.multiply(Pci, similarity_matrix[:, i]), axis = 1), np.divide(Pc, Pi))
            
            
            #zone non optimisé du cul
            #calcul de la double somme s(C)
            sC = np.empty(Nc)
            for C in range(Nc):
                total = 0
                for k in range(nbData):
                    tmpTot = 0
                    for l in range(nbData):
                        factor = Pi / Pc[C]
                        tmpTot += (Pci[C,k] * factor) * (Pci[C,l] * factor) * similarity_matrix[k,l]
                    total += tmpTot
                sC[C] = total
            #fin de la zone non optimisé du cul
            
            # calcul la première ligne de la boucle for du pseudo_code 
            newPci = np.multiply(np.exp(np.divide(np.subtract(np.multiply(sCi, 2), sC), 1/T)), Pc)
            
            Pci[:, i] = newPci
            
            
            #calcul la deuxième ligne de la boucle for du pseudo-code
            newPci = np.divide(Pci[:,i], np.sum(Pci[:, i]))
            
            Pci[:, i] = newPci
        
        #test si toute nos valeurs sont inférieur à epsilon
        if np.all(np.less_equal(np.abs(np.subtract(Pci, lastPci)), epsilon)):
                break

    return Pci



s = np.genfromtxt("data/DS1/simil_ds1.d")

#bug : l'algo ne converge pas
#bug 2 : warning python ligne 50 -> RuntimeWarning: invalid value encountered in true_divide
res = gen_cluster(s, 3)                

for i in range(3):
    print(res[:, i])
    print()

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