import numpy as np
import matplotlib.pyplot as plt
import time

# N : number of data
# Nc : requested number of cluster
# epsilon : convergence parameter, very weak, less than 10^-4
# T : tradeoff parameter // tradeoff between energy and entropy like terms


#génére P(C|i) à m = 0
def init_prob(nbData, nbCluster):
    t = np.random.sample( (nbCluster, nbData) )
    return np.apply_along_axis(lambda a : np.divide(a, np.sum(a)), axis=1, arr=t)

def gen_cluster(similarity_matrix, Nc, T=2, epsilon=0.0001, debug=False):
    nbData = np.shape(similarity_matrix)[0]
    Pci = init_prob(nbData, Nc)

    #calcul p(i), comme les données ont toute la même proba, c'est juste i/N
    Pi = 1/nbData
    
    m = 0

    while True:
        lastPci = np.copy(Pci)
        
        for i in range(nbData):
            
            #calcul P(C) pour tout C
            Pc = np.sum(Pci, axis = 1) * Pi

            bayesFactor = np.divide(Pi, Pc)
            
            #calcul s^(m)(C;i) pour tout C
            sCi = np.multiply(np.sum(np.multiply(Pci, similarity_matrix[:, i]), axis = 1), bayesFactor)

            
            squareFactor = np.power(bayesFactor, 2)
            
            #calcul s^(m)(C) pour tous C, pour tous l appartenant à range(nbData)
            sC = np.zeros(Nc)
            for k in range(nbData):
                tmp =  np.multiply(np.multiply(squareFactor,Pci[:,k]), np.multiply(Pci,similarity_matrix[k,:]).T)
                tmp2 =  np.sum(tmp, axis = 0)
                sC = np.add(tmp2, sC)
            
            
            # calcul la première ligne de la boucle for du pseudo_code 
            newPci = np.multiply(np.exp(np.divide(np.subtract(np.multiply(sCi, 2), sC), T)), Pc)
            
            Pci[:, i] = newPci
            
            #calcul la deuxième ligne de la boucle for du pseudo-code
            newPci = np.divide(Pci[:,i], np.sum(Pci[:, i]))
            
            
            Pci[:, i] = newPci
        tmp = np.abs(np.subtract(Pci, lastPci))
        if(debug):
            print("step :", m, "   max error :", np.max(tmp))
        test = np.less_equal(tmp, epsilon)
        
        m+=1
        
        #test si toute nos valeurs sont inférieur à epsilon
        if np.all(test):
                break

    return Pci



s = np.genfromtxt("data/DS1/simil_ds1.d")

res = gen_cluster(s, 3, T = 0.5, epsilon=0.5)                

cluster = []

#for i in range(np.shape(s)[0]):
#    print("data : ", i)
 #   print("cluster trouvé : ", np.argmax(res[:, i]))
  #  print()


#show result data
s = np.loadtxt("data/DS1/scatter_ds1.d")

mapColor = {
    0:'r',
    1:'b',
    2:'g'
}

labelColor = [mapColor[l] for l in s[:,2]]

plt.scatter(s[:,0],s[:,1], c=labelColor)
plt.title("Original")
plt.show()

labelColor2 = [mapColor[l] for l in np.argmax(res, axis=0)]
plt.scatter(s[:,0],s[:,1], c=labelColor2)
plt.title("T=0.5  eps=0.5")
plt.show()

# DS2
print("DS2 test")

#s = np.genfromtxt("data/DS2/simil_ds2.d")
#res = gen_cluster(s, 5, T = 0.3, epsilon=0.01, debug=True)

# Je pense pas que ce soit vraiment la meilleure représentation, mais je sais pas quoi d'autre faire, parce qu'on a pas de scatter_ds2.d :P
# Je sais pas comment on pourrait faire pour vérifier que le nombre de clusters est correct
# Mais au moins faire un scatter plot de la matrice de similarité n'est sûrement pas correct XD
#plt.scatter(s[:,0],s[:,1], c=np.argmax(res, axis=0))
#plt.show()