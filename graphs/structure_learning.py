import sys, os
try:
    thisFilePath = os.path.abspath(__file__)
except NameError:
    print("Error: __file__ is not available. 'thisFilePath' will resolved to os.getcwd().")
    thisFilePath = os.getcwd()  # Use current directory or specify a default

# Based on this file's path, infer this project's directory
projectPath = os.path.normpath(os.path.join(thisFilePath,'..')) # Move up to project root


if projectPath not in sys.path:  # Avoid duplicate entries
    sys.path.append(projectPath)

import utils.gcsi_utils as gcsi_utils
import utils.gsl as gsl_utils
import utils.gl.src.gl_models as gl_models

import numpy as np
from scipy.sparse import coo_matrix

import torch
from torch_cluster import knn_graph
from sensing_models.sd_cassi import SignalSubDomain

def extract_graph_signals_from_side_image(side_img, omega_subdomain : SignalSubDomain, n1, n2, L):
    """
    Extract graph signals from side image for each wavelength band in the subdomain.
    
    Parameters
    ----------
        side_img : np.ndarray
            Panchromatic or RGB image of shape (n1, n2) or (n1, n2, 3)
        omega_subdomain : sd_cassi.SignalSubDomain
            Signal subdomain with vertices per wavelength band
        n1, n2, L : int
            Spectral image dimensions
        
    Returns
    -------
        signals : np.ndarray
            Stacked signals from all wavelength bands, shape (total_vertices, 1)
        spatial_coords : np.ndarray
            Spatial coordinates for each signal, shape (total_vertices, 2) with [row, col]
    """
    wavelength_coords = omega_subdomain.get_wavelength_coords()
    
    signals_list = []
    spatial_coords_list = []
    
    for band_idx in wavelength_coords:
        vertex_indices = omega_subdomain.vertices[band_idx]
        row, col, _ = np.unravel_index(vertex_indices, (n1, n2, L), order='F')
        
        # Extract signal values at these spatial locations
        band_signal = side_img[row, col].reshape(-1, 1)
        band_coords = np.column_stack((row, col))
        
        signals_list.append(band_signal)
        spatial_coords_list.append(band_coords)
    
    signals = np.vstack(signals_list)
    spatial_coords = np.vstack(spatial_coords_list)
    
    return signals, spatial_coords


def build_knn_edges_per_band(signals, spatial_coords, omega_subdomain : SignalSubDomain, k):
    """
    Build kNN edge sets using joint intensity-spatial features for each band.
    
    Parameters
    ----------
        signals : np.ndarray
            Signal values, shape (total_vertices, 1)
        spatial_coords : np.ndarray
            Spatial coordinates, shape (total_vertices, 2)
        omega_subdomain : sd_cassi.SignalSubDomain
            Subdomain to determine band boundaries
        k : int
            Number of nearest neighbors
        
    Returns
    -------
        edge_set : np.ndarray
            Combined edge indices across all bands, shape (2, num_edges)
    """
    wavelength_coords = omega_subdomain.get_wavelength_coords()
    edge_sets = []
    vertex_offset = 0
    
    for band_idx in wavelength_coords:
        n_vertices = len(omega_subdomain.vertices[band_idx])
        
        # Extract features for this band
        band_signals = signals[vertex_offset:vertex_offset + n_vertices]
        band_coords = spatial_coords[vertex_offset:vertex_offset + n_vertices]
        
        # Normalize features
        signal_scale = band_signals.mean()
        height = band_coords[0].max()-band_coords[0].min()
        width = band_coords[1].max()-band_coords[1].min()
        spatial_scale = np.sqrt(height**2+width**2)/2
        
        normalized_signals = band_signals / signal_scale if signal_scale > 0 else band_signals
        normalized_coords = band_coords / spatial_scale if spatial_scale > 0 else band_coords
        
        # Joint feature space: [normalized_intensity, normalized_row, normalized_col]
        joint_features = np.hstack((normalized_signals, normalized_coords))
        
        # Build kNN edges for this band
        band_edges = knn_graph(torch.Tensor(joint_features), k, loop=False).numpy()
        
        # Offset indices to global indexing
        band_edges += vertex_offset
        edge_sets.append(band_edges)
        
        vertex_offset += n_vertices
    
    return np.hstack(edge_sets)


def build_knn_graph_adj_mtrx(side_img, omega_subdomain : SignalSubDomain, spectral_img_shape, num_neigs=33):
    """
    Construct kNN graph adjacency matrix from side image.
    
    Parameters
    ----------
        side_img : np.ndarray
            Side information image
        omega_subdomain : sd_cassi.SignalSubDomain
            Signal subdomain
        spectral_img_shape : tuple[int, int, int]
            Shape (n1, n2, L) of spectral image
        num_neigs : int
            Number of nearest neighbors
        
    Returns
    -------
        A_G : scipy.sparse matrix
            Symmetric adjacency matrix
    """
    n1, n2, L = spectral_img_shape
    
    # Step 1: Extract signals and coordinates
    signals, spatial_coords = extract_graph_signals_from_side_image(
        side_img, omega_subdomain, n1, n2, L
    )
    
    # Step 2: Build kNN edges using joint features
    edge_set = build_knn_edges_per_band(
        signals, spatial_coords, omega_subdomain, num_neigs
    ) # 2D matrix, where rows are vertex pairs or edges.
    
    # Step 3: Construct symmetric adjacency matrix
    n_vertices, m = signals.shape
    A_G = coo_matrix(
        (np.ones(edge_set.shape[1]), (edge_set[0], edge_set[1])),
        shape=(n_vertices, n_vertices)
    )
    A_G = A_G.maximum(A_G.T)
    
    return A_G

#def build_knn_graph_adj_mtrx(pan_img, Omega_k, spectral_img_shape : tuple[int,int,int], num_neigs = 33):
#    """ Construct knn graph adjanceny matrix """
#    n1, n2, L = spectral_img_shape
#    # Construct graph signal z_k on Omega_k and learn from it a knn graph on Omega_k.
#    z_k, edge_set_k = gcsi_utils.preprocess_side_image_for_graph_learning(pan_img,Omega_k,n1,n2,L,num_neigs)

#    # Construct knn graph adjanceny matrix based on the above edge set.
#    n = len(z_k)
#    A_G = coo_matrix((np.ones(edge_set_k[0].shape),(edge_set_k[0],edge_set_k[1])),shape=(n,n))
#    A_G = A_G.maximum(A_G.T)
#    return A_G #or W_G
# -------- Definition of Kalofolias graph structure learning method -------------------------------          
def build_kalofolias_graph_adj_mtrx(side_img, omega_k : SignalSubDomain, spectral_img_shape : tuple[int,int,int], num_neigs = 33):
    n1, n2, L = spectral_img_shape

    # Step 1: Extract signals and coordinates. These are graphs signals on omega_k
    signals, spatial_coords = extract_graph_signals_from_side_image(
        side_img, omega_k, n1, n2, L
    )
    n_vertices, m = signals.shape

    # Step 2: Build kNN edges using joint features
    edge_set_k = build_knn_edges_per_band(
        signals, spatial_coords, omega_k, num_neigs
    )

    assert edge_set_k.ravel().max() <= n_vertices and edge_set_k.shape[0] == 2

    # Step 3: Build sparsified distance matrix and run Kalofolias structure learning method on it
    # - Compute Euclidean distance between vertex pairs
    i = edge_set_k[0,:].flatten()
    j = edge_set_k[1,:].flatten()
    dist_values = np.sum((signals[i,:] - signals[j,:])**2,axis=1).flatten()
    dist_mtx = coo_matrix((dist_values,(i,j)),shape=(n_vertices,n_vertices)).tocsr()
    # - Ensure distance matrix is symmetric
    dist_mtx = dist_mtx.maximum(dist_mtx.transpose())

    params = {}
    params['verbosity'] = 3
    params['maxit'] = 1000
    params['nargout'] = 1
    a, b, theta = 1, 1, 100
    #theta = gl.estimate_theta(D,q)
    W_G = gl_models.gsp_learn_graph_log_degrees(theta*dist_mtx,a,b,params)
    #W[W<1e-5] = 0 
    return W_G

# Definition of Rank Order Path Graph Structure learning  Method ------------------------------------------

import numpy as np
from scipy.sparse import diags, eye, kron, block_diag, csr_matrix

def path_adjacency(n: int) -> csr_matrix:
    """
    Adjacency of a 1-D path graph P_n (free/Dirichlet boundaries):
    nodes 0..n-1 with edges (i, i+1).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    main = np.zeros(n)
    off = np.ones(n-1)
    A = diags([off, off], [-1, 1], shape=(n, n), format='csr')
    return A

def rank_order_path_adjacency(n,rinv) -> csr_matrix:
    P = eye(n, format='csr')
    P = P[rinv.flatten(),:]
    A_G = path_adjacency(n)

    return P.T @ (A_G @ P)



def build_rop_adj_mtrx_per_band(Z, omega_k : SignalSubDomain):

    wavelength_index = omega_k.get_wavelength_coords() # <=> omega_x0y0.vertices.keys()

    blocks = []
    for l in wavelength_index:
        z_l = omega_k.extract_vertex_features(Z,l)
        rinv = np.argsort(z_l,axis=0)
        n = len(rinv)
        A_G = rank_order_path_adjacency(n,rinv)
        blocks.append(A_G)

    return block_diag(blocks, format='csr')
    

def build_rop_graph_adj_mtrx(pan_img, Omega_k):#DCSDCassiModelObj : DualCameraSDCassiModel, block):
    #x0, y0, height, width = block

    # Omega_tilde_k, Omega_k = DCSDCassiModelObj.sdcassi_obj.get_system_submtx_pair( k = (x0,y0,height,width))

    #pan_img = DCSDCassiModelObj.Z

    W_G = build_rop_adj_mtrx_per_band(pan_img, Omega_k)

    return W_G

# --- Below I am going to begin exploring product graph constructions

def grid_adjacency(nx: int, ny: int, fortran_order: bool = True) -> csr_matrix:
    """
    Adjacency for a 2-D grid (ny by nx) with 4-neighborhood, free boundaries,
    built as the Cartesian product of P_nx and P_ny using Kronecker sums.

    Fortran ordering means we map (i, j) -> i + nx*j, i in [0,nx-1], j in [0,ny-1].
    With that mapping, the Kronecker-sum form is:
        A = kron(I_ny, A_x) + kron(A_y, I_nx)
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny must be >= 1")

    Ax = path_adjacency(nx)
    Ay = path_adjacency(ny)

    Ix = eye(nx, format='csr')
    Iy = eye(ny, format='csr')

    # Fortran (column-major) indexing: i + nx*j
    # Edges along x use kron(Iy, Ax); along y use kron(Ay, Ix)
    A = kron(Iy, Ax, format='csr') + kron(Ay, Ix, format='csr')

    if not fortran_order:
        # If you prefer C-order (row-major) mapping (i,j) -> j + ny*i,
        # just swap the terms (equivalent to reindexing):
        A = kron(Ix, Ay, format='csr') + kron(Ax, Iy, format='csr')

    return A

def block_diag_of_grids(Omega_x0y0, fortran_order: bool = True) -> csr_matrix:
    """
    Given a sequence of square mesh sizes, e.g., sizes=[8, 16, 32],
    build each 2-D grid's adjacency and stack them into a block-diagonal matrix.
    """
    n1 = Omega_x0y0.n1
    n2 = Omega_x0y0.n2
    L = Omega_x0y0.L
    wavelength_coord = Omega_x0y0.get_wavelength_coords()
    blocks = []
    for l in wavelength_coord:
        vertex_indeces = Omega_x0y0.vertices[l]
        coords = np.unravel_index(vertex_indeces,shape=(n1,n2,L),order='F')
        ny = coords[0].max()-coords[0].min()+1
        nx = coords[1].max()-coords[1].min()+1
        A_G_l = grid_adjacency(nx,ny,fortran_order=True)
        blocks.append(A_G_l)

    return block_diag(blocks, format='csr')


