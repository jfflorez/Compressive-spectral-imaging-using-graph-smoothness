
from turtle import width
import numpy as np
from scipy.sparse import coo_matrix, isspmatrix, find
import scipy.io
import matplotlib.pyplot as plt

def reshape_as_column(x):
        return np.reshape(x, (x.size,1), 'F')

class SignalSubDomain:
    def __init__(self,omega_k,n1,n2,L) -> None:
        """ parses a set of coordinates omega_k from a n1-by-n2-ny-L mesh as a dictionary,
        where key provide access to coordinates with the same wavelength coordinate.         
        """
        if len(omega_k.keys()) > L:
            raise ValueError(f'Parameter L : {L} cannot be less than len(omega_k.keys())')
        self.vertices = omega_k
        self.L = L
        self.L_ = len(omega_k.keys())
        self.n1 = n1
        self.n2 = n2

    def get_wavelength_coords(self):
        return self.vertices.keys()

    def get_vertex_coords(self,l):
        # returns vertex coordinate on a n1-by-n2-by-L meshgrid.
        vertex_indices = self.to_array()
        return np.unravel_index(vertex_indices, shape=(self.n1,self.n2,self.L),order='F')
    
    # TODO: Evaluate if this is useful
    def contruct_vertex_features(self,Z,l):

        """Samples Z at vertex coordinates to generate node/vertex features"""

        if not self.n1 == Z.shape[0] and not self.n2 == Z.shape[1]:
            raise ValueError(f'Invalid shape : {Z.shape}. Z must be of shape ({self.n1},{self.n2},x)')
        vertex_indices = self.vertices[l]
        coords = np.unravel_index(vertex_indices,shape=(self.n1,self.n2,self.L),order='F')
        nx = coords[1].max()-coords[1].min()+1
        ny = coords[0].max()-coords[0].min()+1

        return np.reshape(Z[coords[0],coords[1]], shape = (nx*ny,1),order='F')

         
    def to_array(self):
        """ convert vertex indices from dict to linear array representation
            We assume
            self.vertices is a dictionary storing key, pairs,
            where key index the wavelength dimension of an spectral image and
            self.vertices[key] are linear indices at that wavelength giving access to
            patch or sample of values.
        """
        fst_key = list(self.vertices.keys())[0]
        for key in self.vertices: #
            if key > fst_key: 
                out = np.hstack((out,self.vertices[key]))
                #x = np.hstack((x,dataset['X'][np.unravel_index(omega_k[key],(n1,n2,L),order='F')]))
            else: 
                out = self.vertices[key]
                #x = dataset['X'][np.unravel_index(omega_k[key],(n1,n2,L),order='F')]
        return out

class SingleDisperserCassiModel:
    def __init__(self,n1,n2,L,mask,disp_dir) -> None:
        self.n1 = n1
        self.n2 = n2
        self.L = L
        if disp_dir in ['left2right','right2left']:
            self.disp_dir = disp_dir
        else:
            raise NameError("disp_dir must be set as either 'left2right' or 'right2left'")
        # Verify datacube shape
        if (self.n1 == mask.shape[0]) and (self.n2 + self.L - 1 == mask.shape[1]) and (self.L == mask.shape[2]):
            self.mask = mask
        else:
            raise NameError('Invalid shape. T must be 3 dimensional array of shape (n1,n2+L-1,L)')
    def get_spectral_image_shape(self):
        return self.n1, self.n2, self.L
    
    def load_system_mtx(self):
        self.Hmtx = construct_system_mtx(self.mask,self.n1,self.n2,self.L,self.disp_dir)

    def take_simulated_snapshot(self,X,SNR):

        if not (X.shape[0] == self.n1 and X.shape[1] == self.n2 and X.shape[2] == self.L):
            raise NameError('Invalid shape. X must be a 3 dimensional array of shape (n1,n2,L).')

        if not hasattr(self,'Hmtx'):
            self.load_system_mtx()
            print('System matrix Hmtx has been successfully created and added to models attributes.')

        n = self.n1*self.n2*self.L
        m = self.n1*(self.n2 + self.L - 1)
        y = self.Hmtx*np.reshape(X,(n,1),'F')
        
        if SNR != np.inf:
            sigma = np.sqrt(np.std(y,axis=0)**2/(10**(SNR/10)))
            y = y + sigma*np.random.randn(m,1)

        self.Y = np.reshape(y,(self.n1,self.n2 + self.L - 1),'F')
        print('Real coded snapshot Y has been successfully loaded and added to models attributes.')

    def load_real_snapshot(self, Y : np.ndarray):
        if not (Y.shape[0] == self.n1 and Y.shape[1] == self.n2 + self.L - 1):
            raise NameError('Invalid shape. Y must be a 2 dimensional array of shape (n1,n2+L-1).')
        if not hasattr(self,'Hmtx'):
            self.load_system_mtx()
            print("System matrix Hmtx has been successfully created and added to model's attributes.")
        
        self.Y = Y
        print("Real coded snapshot Y has been successfully loaded and added to model's attributes.")
    # TODO: Should we optimize the function below? Is there a better way to implement it?
    def get_system_submtx_pair(self,k):
        """ defines a measurement and signal domain pair (Omega_k_tilde, Omega_k) such that
        Y at Omega_k_tilde indexes a patch of shape (height,width) with topleft corner given by
        y0,x0, and X at Omega_tilde indexes a parallelepiped cube which forms the patch after passing
        through the system matrix Hmtx.
        
        Parameters:
            k : tuple, (x0,y0,height,width)

        Returns:
            omega_k_tilde : index array
            omega_k : dict index array   
        
        """
        x0 = k[0]; y0 = k[1]; height = k[2]; width = k[3]



        #if not (x0 >= 0 and (x0 + width - 1) - 1 <= (self.n2 + self.L - 1) - 1):
        #    raise ValueError("")
        
        x = np.arange(x0, np.min([x0 + width - 1, self.n2 + self.L - 1 - 1]) + 1) # add 1 to include x0 + width - 1
        y = np.arange(y0, np.min([y0 + height - 1, self.n1-1]) + 1)
        xv, yv = np.meshgrid(x, y, indexing='ij')

        # Define linear indices of a patch of shape (height,width) with top left corner given by y0,x0 
        #omega_tilde = g_tilde(reshape_as_column(yv),reshape_as_column(xv)).squeeze()

        omega_k_tilde = np.ravel_multi_index((yv.flatten(),xv.flatten()), (self.n1,self.n2+self.L-1), order='F')

        # Finds indices of datacube X that contribute to formation of Y at omega_tilde

        #omega_ind_tmp = np.argwhere(np.asarray(np.sum(self.Hmtx.tocsr()[omega_k_tilde,:],axis = 0))>0)
        if not hasattr(self,'Hmtx_allones'):
            mask_allones = make_mask_all_ones(self.mask>0)
            self.Hmtx_allones = construct_system_mtx((mask_allones),self.n1,self.n2,self.L,self.disp_dir).tocsr()
        
        
        
        rows = self.Hmtx_allones[omega_k_tilde, :]
        omega_ind_tmp = np.unique(rows.indices)

        #omega_ind_tmp = np.unique(self.Hmtx_allones.tocsr()[omega_k_tilde,:].nonzero()[1])
        
        omega_sub_tmp = np.unravel_index(omega_ind_tmp.flatten(), (self.n1,self.n2,self.L), order = 'F')

        z = np.unique(omega_sub_tmp[2])
        l0 = np.min(z)
        L_s = len(z) # number of bands contributing to formation of Y at omega_tilde 

        #x = np.max(omega_sub_tmp[1]) - np.min(omega_sub_tmp[1]) + 1     
        #if (np.max(omega_sub_tmp[1]) - np.min(omega_sub_tmp[1]) + 1) != (width + L_s - 1):
        #    raise Warning(" X at omega may not be a parallelepiped")

        omega_k = {}
        for l in range(L_s): #:range(l0,L_s): # loop over the bands 

            x0_max = np.max(omega_sub_tmp[1][omega_sub_tmp[2]==z[l]])
            x0_min = np.min(omega_sub_tmp[1][omega_sub_tmp[2]==z[l]])
            y0_max = np.max(omega_sub_tmp[0][omega_sub_tmp[2]==z[l]])
            y0_min = np.min(omega_sub_tmp[0][omega_sub_tmp[2]==z[l]])

            #print('Width :', x0_max - x0_min + 1)
            #print('Height :', y0_max - y0_min + 1)

            x = np.arange(x0_min,x0_max + 1) # add 1 to include x0 + width - 1
            y = np.arange(y0_min,y0_max + 1)

            xv, yv = np.meshgrid(x, y, indexing='ij')
            yv  = yv.flatten()
            xv  = xv.flatten()
            
            omega_k[z[l].item()] = np.ravel_multi_index((yv, xv, z[l]*np.ones(yv.shape).astype(np.int64)),
                                                      (self.n1,self.n2,self.L), order = 'F')

            x0_min = x0_max = y0_min = y0_max = 0
        return omega_k_tilde, SignalSubDomain(omega_k,self.n1,self.n2,self.L)
    
    def get_linear_system(self, block: tuple[int, int, int, int] = None):
        """ 
            Return the CASSI system of linear equations (coefficient matrix and response vector).
            
            For block extraction, also returns the multi-dimensional indices of the subsampled domain.
        """
        if block is None:
            return self.Hmtx, self.Y.ravel(order='F'), None
        
        x0, y0, height, width = block # Validates 4-element tuple
        omega_tilde_k, omega_k = self.get_system_submtx_pair(block)
        # Multi index of the signal values at omega_k
        multi_idx = np.unravel_index(omega_k.to_array(), self.spectral_shape, order='F')

        # Extract CASSI system submatrix
        Hmtx = self.Hmtx.tocsr()[:, omega_k.to_array()]
        Hmtx = Hmtx[omega_tilde_k, :]
        
        # Extract measurement patch
        m = omega_tilde_k.size
        y = self.Y.ravel('F')[omega_tilde_k].reshape((m, 1))


        return Hmtx, y, multi_idx



def construct_system_mtx(mask, n1, n2, L, disp_dir):

    """ constructs a single-disperser coded aperture snapshot spectral imager (CASSI)
    system matrix provided a calibration cube of size n1 by n2 + L - 1 by L

    Parameters:
    -----------
    self: class attributes, n1,n2,L, and mask.
    mask : np.array
        3D matrix of size n1 by n2 + L - 1 by L, where mask[:,:,l] is the transmittance of monochromatic light at wavelength l
    
    Returns:
    --------
    Hmtx: sparse ndarray, shape (n1*(n2+L-1),n1*n2*L)"""

    assert mask.shape[0] == n1
    assert mask.shape[1]-L+1 == n2
    assert mask.shape[2] == L


    g_tilde = lambda y,x: y + n1*x #y + self.n1*(x - 1)
    g = lambda y,x,z: y + n1*x + n1*n2*z #y + self.n1*(x - 1) + self.n1*self.n2*(z - 1)

    for l in range(L):
        nnz_mask_indicator = mask[:,:,l] > 0
        #ind = np.nonzero(nnz_mask_indicator) # ind[0] y coordinate, ind[1] x coordinate
        #y_coord = ind[0]; x_coord = ind[1]  

        ind = np.argwhere(nnz_mask_indicator)
        y_coord = ind[:, 0]; x_coord = ind[:,1] 
            
        #ll = l + 1         

        if l > 0: 
            I = np.concatenate((I,g_tilde(y_coord,x_coord)))

            if disp_dir == 'right2left':
                J = np.concatenate((J,g(y_coord, x_coord - L + l + 1, l)))
            else: # self.disp_dir == 'left2right':
                #J = np.concatenate((J,g(y_coord, x_coord - l + 1, l)))
                J = np.concatenate((J,g(y_coord, x_coord - l, l)))
            
            nnz_vals = np.concatenate((nnz_vals,mask[nnz_mask_indicator,l]))
        else:
            I = g_tilde(y_coord,x_coord)

            if disp_dir == 'right2left':
                J = g(y_coord, x_coord - L + l + 1, l)
            else: # self.disp_dir == 'left2right':
                #J = np.concatenate((J,g(y_coord, x_coord - l + 1, l)))
                J = g(y_coord, x_coord - l, l)                
            nnz_vals = mask[nnz_mask_indicator,l]            
    
    Hmtx = coo_matrix((nnz_vals,(I,J)),shape=(n1*(n2+L-1),n1*n2*L))

    # ## We now check if Hmtx was successfully constructed
    # m = self.n1*(self.n2+self.L-1)
    # for l in range(self.L):
    #    delta = np.zeros((self.n1,self.n2,self.L)) 
    #    # Simulate uniform monochromatic illumination at wavelength l
    #    delta[:,:,l] = 1
    #    delta = np.reshape(delta,(self.n1 * self.n2 * self.L, 1),'F')

    #    condition = np.sum(Hmtx.tocsr()@delta - np.reshape(self.mask[:,:,l],(m,1),'F'),axis=0)==0
    #    if not condition:
    #        raise NameError('For l=',l,', Mask[:,:,self.mask[:,:,l]l] may have unexpected non-zero values.')
    return Hmtx


def make_mask_all_ones(mask):
    """
    For each frame l, find x-extent where ones exist and add 1 to that region.
    
    Parameters:
        mask: 3D array with shape (height, width, depth)
    
    Returns:
        processed_mask: Modified mask with +1 applied to x-extents per frame
    """
    processed_mask = mask.copy()
    
    for l in range(mask.shape[2]):  # iterate over frames
        frame = mask[:, :, l]
        
        # Find where ones exist in this frame
        ones_indices = np.argwhere(frame == 1)
        
        if len(ones_indices) > 0:  # if there are any ones in this frame
            # Find x-extent (min and max x coordinates)
            xmin_l = np.min(ones_indices[:, 1])  # column 1 is x-coordinate
            xmax_l = np.max(ones_indices[:, 1])
            
            # Apply +1 to the x-extent region
            processed_mask[:, xmin_l:xmax_l+1, l] = processed_mask[:, xmin_l:xmax_l+1, l] + 1
    
    return processed_mask




