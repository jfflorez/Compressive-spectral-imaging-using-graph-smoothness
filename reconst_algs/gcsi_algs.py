import numpy as np

import scipy as sp
import scipy.spatial.distance as sp_sd
from scipy.sparse.linalg import spsolve, minres, lsqr, norm, eigsh
from scipy.sparse import coo_matrix, hstack, vstack, save_npz, load_npz, spdiags, eye

def compute_reconst_loss(x, y, H):
    x_flat = x.flatten()
    y_flat = y.flatten()
    r = (y_flat - (H @ x_flat)).flatten()
    return np.dot(r, r) # returns a scalar

def compute_GTV(x, L_G):
    # Ensure x is a 1 dimensional array required by the dot function. 
    x_flat = x.flatten()
    return np.dot(x_flat, L_G @ x_flat) # returns a scalar


def gsm_noiseless_case_estimation(H, y, W_G, params={'tol':1e-5, 'maxiter': 10000}):
    """ 
    Computes the solution to the problem:  
        min_x x^T (L_G x) s.t. H x = y, 
    using the MINRES algorithm, where L_G = diag(sum(W_G,1)) - W_G.
    
    Parameters
    ----------
    H : scipy.sparse.csr_matrix
        Sparse sensing matrix of shape (m,n).
    y : numpy.ndarray
        Measurement vector of shape (m,1).
    W_G : scipy.sparse.csr_matrix
        Sparse adjacency matrix of the graph G.
    params : dict, optional
        Dictionary with MINRES parameters {'tol', 'maxiter'}.
    return_metadata : bool, optional
        If True, returns (x_hat, metadata) tuple with convergence info.
        
    Returns
    -------
    x_hat : numpy.ndarray
        Signal estimate of shape (n,1).
    metadata : dict (only if return_metadata=True)
        Convergence information including status, iterations, and residual.
    """
    
    m, n = H.shape

    # Set up KKT system of linear equations
    L_G = spdiags(np.sum(W_G, axis=1).squeeze(), 0, n, n) - W_G   
    zeros = coo_matrix((m, m), dtype=np.float32).tocsr()
    
    A = vstack([hstack([L_G, H.T]), hstack([H, zeros])]).tocsr()
    
    b = np.vstack((np.zeros((n, 1)), y))

    minres_kwargs, callable_dict = make_minres_kwargs(A, b, params)

    result = minres(A, b, **minres_kwargs)
    x_solution, info = result[0], result[1]
    x_hat, x_init = x_solution[0:n], minres_kwargs['x0'][0:n]

    #x_solution = spsolve(A, b_0)
    #info = 0
    # Collect some stats about the image patch y and the solution metrics for downstream analyses
    # NOTE: downstream code may get affected if the below dicts are not compatible with h5 attribute types
    data_stats = {'y_var' : y.var(), 'y_mean' : y.mean()}
    solution_metrics = {
        'num_iters' : callable_dict['num_iters'],
        'L2_loss_start' : compute_reconst_loss(x_init,y,H),
        'L2_loss_end' : compute_reconst_loss(x_hat,y,H),
        'GTV_start' : compute_GTV(x_init,L_G),
        'GTV_end' : compute_GTV(x_hat,L_G)
        }
    
    
    
    # Decode MINRES info flag
    convergence_status = {
        0: 'converged',
        1: 'max_iter_reached',
        -1: 'illegal_input_or_breakdown'
    }
    
    converged = (info == 0)
    status = convergence_status.get(info, 'unknown')
    
    if not converged:
        print(f'Warning: MINRES {status} (info={info})')
    else:
        print('Solution converged to desired tolerance.')    

    metadata = {
        'converged': converged,
        'minres_info': int(info),
        'status': status,
        'tol': params['tol'],
        'maxiter': params['maxiter']
    }
    metadata.update(data_stats)
    metadata.update(solution_metrics)

    return x_hat, metadata

def gsm_noisy_case_estimation(H, y, W_G, params = {'alpha' : 7, 'tol':1e-4, 'maxiter': 10000}):
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

        L_G = spdiags(np.sum(W_G, axis=1).squeeze(), 0, n, n) - W_G  
        #L_G = L_G + 1e-1*eye(n,n)
        # Uncomment lines below to test normalized graph Laplacian
        #D_inv_sq = sp.sparse.spdiags(1/np.sqrt(np.sum(W_G, axis=1).squeeze()), 0 , n , n )
        #L_G = sp.sparse.spdiags(np.ones((n,1)).squeeze(), 0 , n , n ) - D_inv_sq @ (W_G @ D_inv_sq)

        A = (H.T @ H) + params['alpha']* L_G 
        b = H.T @ y

        minres_kwargs, callable_dict = make_minres_kwargs(H, y, params)

        result = minres(A.tocsr(), b, **minres_kwargs)
        x_hat, info, x_init = result[0], result[1], minres_kwargs['x0']

        # Collect some stats about the image patch y and the solution metrics for downstream analyses
        # NOTE: downstream code may get affected if the below dicts are not compatible with h5 attribute types
        data_stats = {'y_var' : y.var(), 'y_mean' : y.mean()}
        solution_metrics = {
            'num_iters' : callable_dict['num_iters'],
            'L2_loss_start' : compute_reconst_loss(x_init,y,H),
            'L2_loss_end' : compute_reconst_loss(x_hat,y,H),
            'GTV_start' : compute_GTV(x_init,L_G),
            'GTV_end' : compute_GTV(x_hat,L_G)
            }

        # Decode MINRES info flag
        convergence_status = {
            0: 'converged',
            1: 'max_iter_reached',
            -1: 'illegal_input_or_breakdown'
        }
        
        converged = (info == 0)
        status = convergence_status.get(info, 'unknown')
        
        if not converged:
            print(f'Warning: MINRES {status} (info={info})')
        else:
            print('Solution converged to desired tolerance.')
        
        metadata = {
            'converged': converged,
            'minres_info': int(info),
            'status': status,
            'tol': params['tol'],
            'maxiter': params['maxiter']
        }
        metadata.update(data_stats)
        metadata.update(solution_metrics)

        return x_hat, metadata
#def make_minres_kwargs(A, b, params, eig_tol=1e-6, maxiter_eig=200):
def make_minres_kwargs(A, b, params):
    """
    Create keyword arguments for scipy.sparse.linalg.minres that
    capture iteration count, first/last residuals, and a sparse-safe
    condition number estimate with timing.
    """

    callback_dict = {
        #"residual_norms": [],
        "num_iters": 0,
        #"cond_A": None,
        #"cond_est_time": 0.0,
        #"lambda_min": None,
        #"lambda_max": None,
    }

    # ---- Sparse-safe condition number estimation (timed) ----
    #t0 = time.perf_counter()
    #try:
        # Largest eigenvalue
    #    lambda_max = eigsh(
    #        A, k=1, which="LM", tol=eig_tol, maxiter=maxiter_eig, return_eigenvectors=False
    #    )[0]

        # Smallest eigenvalue
    #    lambda_min = eigsh(
    #        A, k=1, which="SM", tol=eig_tol, maxiter=maxiter_eig, return_eigenvectors=False
    #    )[0]

    #    stats["lambda_max"] = lambda_max
    #    stats["lambda_min"] = lambda_min
    #    stats["cond_A"] = lambda_max / lambda_min

    #except Exception:
    #    stats["cond_A"] = np.nan

    #stats["cond_est_time"] = time.perf_counter() - t0
    # ---------------------------------------------------------

    def callback(xk):
        #rk_norm = norm(b - A @ xk)
        #stats["residual_norms"].append(rk_norm)
        callback_dict["num_iters"] += 1
    n = A.shape[1]
    minres_kwargs = {
        "x0": A.T @ b,
        #"x0" : np.zeros((n,1)),
        "rtol": params["tol"],
        "maxiter": params["maxiter"],
        "callback": callback,
    }

    return minres_kwargs, callback_dict
