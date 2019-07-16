from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def plot_pbs(A_pbs, B_pbs, pb_journey = None, plot_zero = False):
    
    pb_dim = A_pbs.shape[1]
    
    pca = PCA(n_components = 2)
    pca.fit(np.concatenate([A_pbs, B_pbs]))
    
    A_pbs, B_pbs = pca.transform(A_pbs), pca.transform(B_pbs)
    
    unis = np.random.uniform(size=A_pbs.shape[0])
    
    colors = ['red', 'blue', 'green', 'black', 'orange', 'purple', 'pink']
    
    for i in range(A_pbs.shape[0]):
        
        color = colors[i]
        
        for p in [A_pbs[i], B_pbs[i]]:
            x, y = p
            
            plt.scatter(x, y, c = color)

    if pb_journey is not None:
        for i, pb in enumerate(pb_journey):
            color = np.array([[1, 0, 0, 0.2 + (i+1)/len(pb_journey)*0.8]])
            pb = pca.transform(np.expand_dims(pb, 0))
            x, y = pb[0]
            plt.scatter(x, y, c=color, marker='s')
    
    if plot_zero:
        z = pca.transform(np.zeros((1, pb_dim)))
        x, y = z[0]
        plt.scatter(x, y, c='black', marker='+')
    
    plt.show()
