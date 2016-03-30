import numpy as np
    
#%%
def delta(r):
    X_r = X_batch[:,r]
    delta = np.eye((d_x))
    non_observable = X_r==99
    delta[non_observable,non_observable]=0
    return np.array(delta)
#%%    
def group_structure(d_alpha, struct, nb_group=3):
    if struct == 'lasso':
        d_G = []
        for i in range(d_alpha):
            G = np.zeros(d_alpha)
            G[i] = 1
            d_G.append(G)
            
    if struct == 'group_lasso':
        # fonction rand pour donner le nombre d'indices dans chaque partition de {1...d_alpha}
        # Exemple: n paritions
        partition = sorted(np.random.choice(d_alpha, nb_group-1, replace = False))

        ## shuffle les indices
        shuffled_indices = np.random.choice(d_alpha, d_alpha, replace = False)
        
        ## Création des groupes
        x = []
        for i, item in enumerate(partition):
            if i == 0:
                x.append(shuffled_indices[:item])
            elif i == len(partition) - 1:
                x.append(shuffled_indices[partition[i-1]:item])
                x.append(shuffled_indices[item:])
            else:
                x.append(shuffled_indices[partition[i-1]: item])
        d_G = []
        for i, item in enumerate(x):
            G = np.zeros(d_alpha)
            G[item] = 1
            d_G.append(G)
            
    return d_G
#%%
def min_alpha(x_O, D_O, alpha, eta, d_alpha, T_alpha, epsilon, d_G):
    import cvxopt
    ## Minimisation on alpha
    for t in range(T_alpha):
    
        ## Compute z
        norm_G_alpha = []
    
        for j in d_G:
            norm_G_alpha.append(np.linalg.norm(j * alpha, ord =2))
        coef = (np.linalg.norm(norm_G_alpha, ord = eta) ** (eta - 1))
    
        norm_G_alpha = np.array(norm_G_alpha)
        norm_G_alpha = np.power(norm_G_alpha, 2 - eta)
        z = norm_G_alpha * min(coef,10000)
        
        for i, item in enumerate(z):
            z[i] = max(item, epsilon)            #MAX TRANSFO EN MIN
    
        
        ## Compute alpha
        ksi = []
        for j in range(d_alpha):
            coef_ksi = 0.0
            for i, item in enumerate(z):
                coef_ksi += (d_G[i][j])**2 / item 
            ksi.append(coef_ksi)
        
        ### Pour résoudre le pb de progammation quadratique, on utilise cvxopt
        # P = kappa * diag(ksi) + D_O.T.dot(D_O)
        # Q = - D_O.T.dot(x_O)
        P = kappa * np.diag(ksi) + D_O.T.dot(D_O)
        P = cvxopt.matrix(P)
    
        q = - D_O.T.dot(x_O)
        q = cvxopt.matrix(q)
    
        sol = cvxopt.solvers.qp(P,q)
        alpha = np.ravel(sol['x'])
    return alpha
    
#%%
def dictionary(C,B,E,T_D, alpha,D,d_x,d_alpha,R):
       
    #NOTE ici alpha est la concatenation des vecteurs alphas updatés précédement, pour chaque ligne du batch (R lignes). C'est une matric de taille (d_alpha,R)
    
    for t in range(T_D):
        for j in range(d_alpha):
            #compute e_j
            sum_r=0
            for r in range(int(R)) :  
                sum_r += np.dot(np.dot(delta(r),D),alpha[:,r])*alpha[j,r]
            e_j_temp = E[:,j] + 1/R * sum_r
            #compute u_j by solving linear system
            C_j = C[:,:,j]
            u = np.linalg.solve(C_j, B[:,j]-e_j_temp+np.dot(C_j,D[:,j]))
            #u = np.linalg.lstsq(C_j, B[:,j]-e_j_temp+np.dot(C_j,D[:,j]))[0]
            #contraints u_j to keep only positive coefficients
            u[u<0]=0
            #compute d_j by projection of u on the good space
            D[:,j] = u / max(np.linalg.norm(u),1)
            
    return D    

#%%
def proj(matrice):
    #Projette 

#%%   
def OSDL(X,D0,T,R,Group_struct,rho,kapa,eta,epsilon = 0.00001,T_alpha = 5,T_D = 5):
    d_x, N = X.shape
    C = np.zeros((d_x,d_x,d_alpha))
    B = np.zeros((d_x,d_alpha))
    E = np.zeros((d_x,d_alpha))
    alpha = np.zeros((d_alpha,R))
    D = D0[:]
    
    for t in range(T):
        print "iteration N° : ",t
        #Draw samples for mini-batches
        X_batch = X[:,t*R:(t+1)*R] #Dans notre cas on coupe artificiellement le dataset en batchs
        
        #----------------------------------------------------------------------------------------        
        #Compute alphas_r representations
        for r in range(R):
            x_temp = list(X_batch[:,r])
            ## Save indexes of observed values
            indices = [i for i, item in enumerate(x_temp) if item != 99]
            #remove missing values of first row of R_train
            x_O = [a for a in x_temp if a != 99]
            # save the observed rows of D
            D_O = D[np.array(indices),:].copy() 
            #Compute the r column of alpha
            alpha[:,r] = min_alpha(x_O, D_O, alpha[:,r], eta, d_alpha, T_alpha, epsilon, d_G)
        
        
        #----------------------------------------------------------------------------------------    
        #Update the statistics
        gamma=(1-1/float(t+1))**rho
        
        #Update B
        sum_b = np.zeros((d_x,d_alpha))
       
        for r in range(R):
            sum_b += np.dot(np.expand_dims(np.dot(delta(r),X_batch[:,r]),axis=1),np.expand_dims(alpha[:,r],axis=1).T)
        B = gamma*B + 1/float(R)*sum_b
        
        #Update all the C_j matrices and e_j vectors
        sum_c = np.zeros((d_x,d_x))
        
        for j in range(d_alpha):
            C_j = C[:,:,j]
            E[:,j] = gamma*-E[:,j] #partial update on e_j
            
            for r in range(R):
                sum_c += delta(r)*alpha[j,r]**2
            
            C_j = gamma*C_j + 1/float(R)*sum_c
            C[:,:,j] = C_j
        #----------------------------------------------------------------------------------------
        #Update dictionnary 
        D = dictionary(C,B,E,T_D, alpha,D,d_x,d_alpha,R)
        
        #----------------------------------------------------------------------------------------
        #Finish update on E
        sum_e = np.zeros(d_x)
        for j in range(d_alpha):
            e_j = E[:,j]
            for r in range(R):
                sum_e += np.dot(np.dot(delta(r),D),alpha[:,r])*alpha[j,r]
                
            e_j = e_j + 1/R*sum_e
            E[:,j] = e_j #final update
        
    
    return D    
    


#%% Lecture du Jester dataset
import pandas as pd
xl = pd.ExcelFile('jester-data-1.xls')
df = xl.parse('jester-data-1-new', header = None)
del df[0]
ratings = np.array(df)
#%%
R_train = ratings[:10000]
R_val = ratings[10000:20000]
R_test = ratings[20000:]
#%%
### Initlialize of parameters 

d_alpha = 100
d_x = 100
alpha = np.random.randn(d_alpha)
T_alpha = 5
T_D = 5
T = 50
R = 8
epsilon = 10**(-5)
rho = 32
eta = 0.5
kappa = 1. / (2**10)
#d_G = np.ones(d_alpha)
d_G = group_structure(d_alpha, struct='lasso', nb_group=10)
D0 = np.random.rand(d_x,d_alpha)
#%%
X = R_train.T
D = OSDL(X,D0,T,R,d_G,rho,kappa,eta,epsilon,T_alpha,T_D)