import numpy as np

import scipy as sp
import scipy.spatial.distance as sp_sd
from scipy.sparse import coo_matrix, hstack, vstack, save_npz, load_npz

import torch
from torch_cluster import knn_graph
import matplotlib.pyplot as plt

import utils.gl.src.gl_models as gl

import os

def preprocess_side_image_for_graph_learning(Z,Omega_x0y0,n1,n2,L,k_):
    """
    Parameters:
        Z : np.ndarray, image of shape (n1,n2) or (n1,n2,3) captured by a panchromatic or color side info camera.

        Omega_x0y0 : sd_cassi.SignalSubDomain, signal subdomain associated with measurement patch x0y0 on which 
        graph must be inferred from signals suitably generated from the side image Z.
        
    Returns:
        z_x0y0 : 
        edge_const_set:
    """
    l0 = list(Omega_x0y0.vertices.keys())[0]
    #for l in Omega_x0y0.vertices:
    wavelength_coord = Omega_x0y0.get_wavelength_coords()
    for l in wavelength_coord: # loop over wavelength index 

        # Vertices associated with l-th band
        vertex_indices = Omega_x0y0.vertices[l]
        coords = np.unravel_index(vertex_indices,(n1,n2,L),order='F')
        
        print(f'{l}, xmin: {coords[1].min()}, xmax: {coords[1].max()}')
        #print(f'ymin: {coords[0].min()}, ymax: {coords[0].max()}')
        height = coords[0].max()-coords[0].min()
        width = coords[1].max()-coords[1].min()
        hs = np.sqrt(height**2+width**2)/2
        N_l = coords[0].size
        z_l = np.reshape(Z[coords[0],coords[1]],(N_l,1), order = 'F')

        hr = z_l.mean()
#        if knn_edge_cons:
        joint_features = np.concatenate((z_l/hr,np.reshape(coords[0],(N_l,1), order = 'F')/hs,np.reshape(coords[1],(N_l,1), order = 'F')/hs),axis=1)
        edge_set_l = knn_graph(torch.Tensor(joint_features), k_, loop=False)
        #edge_index_l = edge_index_l.numpy()

        #r_inv = np.argsort(z_l,axis=0)
        #z_l[r_inv.flatten()] = np.reshape(np.arange(0,N_l),(N_l,1))
        if l > l0:
            # Locate indices in the l-th diagonal block by adding z_x0y0.shape[0]
            edge_set_l = edge_set_l + z_x0y0.shape[0]
            edge_set_x0y0 = torch.hstack((edge_set_x0y0, edge_set_l))
            z_x0y0 = np.vstack((z_x0y0, z_l))            
        else:
            edge_set_x0y0 = edge_set_l
            z_x0y0 = z_l  

        
        #width = np.max(coords[1])-  np.min(coords[1]) + 1
        #height = np.max(coords[0])-  np.min(coords[0]) + 1
        #plt.figure()
        #plt.imshow(np.reshape(z_l , (height,width), order = 'F'))        
    return z_x0y0, edge_set_x0y0.numpy()


def generate_sd_cassi_calibration_cube(t,n1,n2,L):

    np.random.seed(0)

    mask = np.zeros((n1,n2 + L - 1,L))
    mask_tmp = np.random.rand(n1,n2) > t
    multi_idx = np.argwhere(mask_tmp)
    for l in range(L):
        mask[multi_idx[:,0],multi_idx[:,1],l] = 1
        multi_idx[:,1] = multi_idx[:,1] + 1
        #plt.figure()
        #plt.imshow(mask[:,:,l]) 

    return mask

def construct_graph_on_omega_x0y0(z_x0y0,edge_set_x0y0, q):
    """"""

    K = z_x0y0.shape[1]
    n = z_x0y0.shape[0]
    params = {}
    #params['edge_mask'] = sp.sparse.coo_matrix((np.ones((edge_set_x0y0[0,:].shape)).squeeze(), 
    #                                            (edge_set_x0y0[0,:],edge_set_x0y0[1,:])), shape= (n,n)) 
    #params['edge_mask'] = params['edge_mask'] + params['edge_mask'].T


    dist = lambda z, edge_set: np.sum((z[edge_set[0,:].squeeze(),:] - z[edge_set[1,:].squeeze(),:])**2,axis=1)
    D = sp.sparse.coo_matrix((dist(z_x0y0,edge_set_x0y0), (edge_set_x0y0[0,:],edge_set_x0y0[1,:])), shape=(n,n)) #np.zeros(Z.shape) #np.reshape(np.array([1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0]),(16,1))
    D = D.maximum(D.transpose())
    #D = D.to_csr()
    #z = (np.sqrt(d)*sp_sd.pdist(X, 'chebyshev'))**2
    #Z = sp_sd.squareform(z) # turns the condensed form into a n by n distance matrix


    
    #params['w_0'] = np.zeros((m,m))
    #params['c'] = 1
    #if knn_edge_cons:
    #    edge_index = knn_graph(torch.Tensor(X), k_, loop=False)
    #    edge_index = edge_index.numpy()
    #    params['fix_zeros'] = True
    #    params['edge_mask'] = sp.sparse.coo_matrix((np.ones((edge_index[0].shape)).squeeze(), (edge_index[0],edge_index[1])), shape=Z.shape) #np.zeros(Z.shape) #np.reshape(np.array([1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0]),(16,1))
    #    params['edge_mask'] = params['edge_mask'] + params['edge_mask'].T
    params = {}
    params['verbosity'] = 3
    params['maxit'] = 1000
    params['nargout'] = 1

    a = 1
    b = 1
    #theta = gl.estimate_theta(D,q)
    theta = 100
    W = gl.gsp_learn_graph_log_degrees(theta*D.tocsr(),a,b,params)
    #W[W<1e-5] = 0 
    return W, theta

def save_graph_on_omega_x0y0(hsdc_name, W, x0,y0):

    if not os.path.isdir('precomputed_graphs'):
        os.mkdir('precomputed_graphs')

    graph_name = lambda name, x0, y0: hsdc_name + '_graph_y0_' + str(y0) + '_x0_' + str(x0)

    path = os.path.join(os.getcwd(),'precomputed_graphs')
    path = os.path.join(path, graph_name(hsdc_name,x0,y0))

    save_npz(path,W)

def load_graph_on_omega_x0y0(hsdc_name, x0,y0):

    graph_name = lambda name, x0, y0: hsdc_name + '_graph_y0_' + str(y0) + '_x0_' + str(x0) + '.npz'

    path = os.path.join(os.getcwd(),'precomputed_graphs')
    path = os.path.join(path, graph_name(hsdc_name,x0,y0))

    #if os.path.isdir(path):
    W = load_npz(path)

    return W



def psnr(X,X_hat):

    L = X.shape[2]
    psnr_value = np.zeros((L,1))
    for l in range(L):
        mse_l = np.nanmean((X[:,:,l] - X_hat[:,:,l]).flatten()**2,)
        psnr_value[l] = 10*np.log10(np.max(X[:,:,l].flatten())**2/mse_l)
    return np.mean(psnr_value,axis=0)
