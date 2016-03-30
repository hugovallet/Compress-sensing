
import numpy as np
   
    
def dictionary2(C,B,e_j,T_D, alpha_mini_batch,D,M_mini_batch,d_x,d_alpha,R):
       
    #Initialization
    
    for t in range(T_D):
        for j in range(d_alpha):
            #compute e_j
            j=0
            R=2
            alpha_mini_batch_j = alpha_mini_batch[j,:]
            alpha_mini_batch_j = np.expand_dims(alpha_mini_batch_j,axis=1) #used for broadcasting alpha_mini_batch size in the following line
            e_new_j = e_j[:,j] + 1/R * np.sum( (np.dot(D,alpha_mini_batch) * (alpha_mini_batch_j*np.ones(d_x)) )  *   M_mini_batch,axis=1)
            #compute u_j
            u = (B[:,j] - e_new_j) / C[:,j] + D[:,j]
            #contraints u_j to keep only positive coefficients
            u[u<0]=0
            #compute d_j by projection of u on the good space
            D[:,j] = u / np.linalg.norm(u)
            
    return D
    
#%%
def delta(r):
    X_r = X[:,r]
    delta = np.eye((d_x))
    non_observable = X_r==99
    delta[non_observable,non_observable]=0
    return np.array(delta)
    

def dictionary(C,B,e_j,T_D, alpha,D,d_x,d_alpha,R):
       
    #Initialization
    
    for t in range(T_D):
        for j in range(d_alpha):
            #compute e_j
            sum_r=0
            for r in range(int(R)) :  
                sum_r += np.dot(np.dot(delta(r),D),alpha[:,r])*alpha[j,r]
            e_j_temp = e_j[:,j] + 1/R * sum_r
            #compute u_j by solving linear system
            C_j = C[:,:,j]
            u = np.linalg.lstsq(C_j,B[:,j]-e_j_temp+np.dot(C_j,D[:,j]))[0]
            #contraints u_j to keep only positive coefficients
            u[u<0]=0
            #compute d_j by projection of u on the good space
            D[:,j] = u / max(np.linalg.norm(u),1)
            
    return D    

#%%   
def OSDL(X,D0,T,R,Group_struct,rho,kapa,eta,d,epsilon = 0.00001,T_alpha = 5,T_D = 5):
    d_x, N = X.shape
    d_alpha = 31
    C = np.zeros((d_x,d_alpha))
    B = np.zeros((d_x,d_alpha))
    E = np.zeros((d_x,d_alpha))
    alpha = np.zeros((d_alpha,R))
    
    for t in range(T):
        #Draw samples for mini-batches
        X_batch = X[:,t*R:(t+1)*R] #Dans notre cas on coupe artificiellement le dataset en batchs
        
        #Compute alphas_r representations
        for r in range(R):
            x_temp = list(X_batch[:,r])
            ## Save indexes of observed values
            indices = [i for i, item in enumerate(x) if item != 99]
            #remove missing values of first row of R_train
            x_O = [a for a in x_temp if a != 99]
            # save the observed rows of D
            D_O = D[np.array(indices),:].copy() 
            #Compute the r column of alpha
            alpha[:,r] = min_alpha(x_O, D_O, alpha[:,r], eta, d_alpha, T_alpha, epsilon, d_G)
            
        #Update the statistics
        gamma=(1-1/float(t))^rho
        
        sum_c = 0
        sum_b = 0
        sum_e = 0
        for r in range(R):
            sum_c += delta(r)XXX
            sum_b += np.dot(np.expand_dims(np.dot(delta(r),X_batch[:,r]),axis=1),np.expand_dims(alpha[:,r],axis=1).T)
            sum_e +=
            
        C = gamma*C + 1/float(R)*sum_c
        B = gamma*B + 1/float(R)*sum_b
        E = gamma*E + 1/float(R)*sum_e
        
        #Update dictionnary 
        D = dictionary(C,B,e_j,T_D, alpha,D,d_x,d_alpha,R)
        
        #Finish update on E
        E = E + XXXXXXXXXXX
        
    
    return D    
    
#%%
    
d_x = 6
d_alpha = 4
R = 3.0

X = np.random.rand(6,2*R)
X[0,0]=X[0,1]=X[5,1]=99
X[3,2]=99
M_mini_batch = X!=99

C = np.ones((d_x,d_x,d_alpha))
e_j = np.zeros((d_x,d_alpha))
B = np.ones((d_x,d_alpha))
D = np.zeros((d_x,d_alpha))

alpha = np.random.rand(d_alpha,R)

T_D=5

dictionary(C,B,e_j,T_D, alpha,D,d_x,d_alpha,R)