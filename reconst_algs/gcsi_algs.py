import numpy as np

import scipy as sp
import scipy.spatial.distance as sp_sd
from scipy.sparse.linalg import spsolve, minres, lsqr
from scipy.sparse import coo_matrix, hstack, vstack, save_npz, load_npz



def gsm_noiseless_case_estimation(H, y, W_G, params = {'tol':1e-5, 'maxiter': 10000}):
        """ computes the solution to the problem:  
                    min_x x^T (L_G x) s.t. H x = y (1), 
            using the algorithm MINRES, where L_G = diag(sum(W_G,1)) - W_G.
            
            Parameters:
                H : scipy.sparse.csr_matrix,  sparse sensing matrix of shape (m,n).
                y : numpy.ndarray, measurement vector of shape (m,n).
                W_G: scipy.sparse.csr_matrix, sparse adjacency matrix of the graph G, on which x s.t. H x = y is assumed to be smoothest.     

                params (optional): dict, dictionary with the paramters of the algorithm MINRES.

            Returns:
                x_hat : numpy.ndarray, signal estimate of shape (n,1)

        """

        m, n = H.shape
        L_G = sp.sparse.spdiags(np.sum(W_G, axis=1).squeeze(), 0 , n , n ) - W_G

        zeros = sp.sparse.coo_matrix((m, m), dtype=np.float32).tocsr()
        A = vstack( [hstack([L_G,H.T]), hstack([H,zeros])] )
        zeros_b = np.zeros((n,1))
        b_0 = vstack([zeros_b, y])
        x_hat = minres(A.tocsr(), b_0.toarray(), rtol=params['tol'], maxiter=params['maxiter'])
        #x_hat = lsqr(A.tocsr(), b_0.toarray(),iter_lim=params['maxiter'])
        if x_hat[1] == 0: # Uncoment this line to try out lsqr x_hat[2] < params['maxiter']: #
            print('Solution x_hat converged to the desired tolerance within the maximum number of iterations.')
        else: 
            #raise Warning('Either solution reached the maximum number of iterations or there was an illegal input')
            print('Warning: Either solution reached the maximum number of iterations or there was an illegal input')
        
        return x_hat[0][0:n]

def gsm_noisy_case_estimation(H, y, W_G, alpha, params = {'tol':1e-4, 'maxiter': 10000}):
        """ computes the solution to the problem:  
                    min_x alpha * (x^T (L_G x)) + || H x - y ||_2^2, 
            using the algorithm MINRES, where L_G = diag(sum(W_G,1)) - W_G.
            
            Parameters:
                H : scipy.sparse.csr_matrix,  sparse sensing matrix of shape (m,n).
                y : numpy.ndarray, measurement vector of shape (m,n).
                W_G: scipy.sparse.csr_matrix, sparse adjacency matrix of the graph G, on which x s.t. H x = y is assumed to be smoothest. 
                alpha: float, regularization paramter    

                params (optional): dict, dictionary with the paramters of the algorithm MINRES.

            Returns:
                x_hat : numpy.ndarray, signal estimate of shape (n,1)

        """
        if not 'maxiter' in params:
             params = {'maxiter': 1000}
        m, n = H.shape

        tau = 0.5*np.min(np.sum(W_G, axis=1).squeeze())
        D_inv_sq = sp.sparse.spdiags(1/np.sqrt(np.sum(W_G, axis=1).squeeze()), 0 , n , n )
        L_G = sp.sparse.spdiags(np.ones((n,1)).squeeze(), 0 , n , n ) - D_inv_sq @ (W_G @ D_inv_sq)

        A = (H.T @ H) + alpha * L_G 
        b_0 = H.T @ y
        x_hat = minres(A.tocsr(), b_0, rtol=params['tol'], maxiter=params['maxiter'])

        if x_hat[1] == 0:
            print('Solution x_hat converged to the desired tolerance within the maximum number of iterations.')
        else: 
            #raise Warning('Either solution reached the maximum number of iterations or there was an illegal input')
            print(' Warning: Either solution reached the maximum number of iterations or there was an illegal input')
        
        return x_hat[0]