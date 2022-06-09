
from turtle import width
import numpy as np
from scipy.sparse import coo_matrix, isspmatrix, find
import scipy.io
import matplotlib.pyplot as plt

def reshape_as_column(x):
        return np.reshape(x, (x.size,1), 'F')

class SignalSubDomain:
    def __init__(self,omega_k) -> None:
        self.vertices = omega_k
        self.L_ = len(omega_k.keys())

    
    def to_array(self):
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
    
    def construct_system_mtx(self):
        """ constructs a single-disperser coded aperture snapshot spectral imager (CASSI)
        system matrix provided a calibration cube of size n1 by n2 + L - 1 by L
        Parameters:
        self: class attributes, n1,n2,L, and mask.
        
        Returns:
        Hmtx: sparse ndarray, shape (n1*(n2+L-1),n1*n2*L)"""
            
        g_tilde = lambda y,x: y + self.n1*x #y + self.n1*(x - 1)
        g = lambda y,x,z: y + self.n1*x + self.n1*self.n2*z #y + self.n1*(x - 1) + self.n1*self.n2*(z - 1)
    
        for l in range(self.L):
            nnz_mask_indicator = self.mask[:,:,l] > 0
            ind = np.nonzero(nnz_mask_indicator) # ind[0] y coordinate, ind[1] x coordinate
            y_coord = ind[0]; x_coord = ind[1]    
            #ll = l + 1         

            if l > 0: 
                I = np.concatenate((I,g_tilde(y_coord,x_coord)))

                if self.disp_dir == 'right2left':
                    J = np.concatenate((J,g(y_coord, x_coord - self.L + l + 1, l)))
                else: # self.disp_dir == 'left2right':
                    #J = np.concatenate((J,g(y_coord, x_coord - l + 1, l)))
                    J = np.concatenate((J,g(y_coord, x_coord - l, l)))
                
                nnz_vals = np.concatenate((nnz_vals,self.mask[nnz_mask_indicator,l]))
            else:
                I = g_tilde(y_coord,x_coord)

                if self.disp_dir == 'right2left':
                    J = g(y_coord, x_coord - self.L + l + 1, l)
                else: # self.disp_dir == 'left2right':
                    #J = np.concatenate((J,g(y_coord, x_coord - l + 1, l)))
                    J = g(y_coord, x_coord - l, l)                
                nnz_vals = self.mask[nnz_mask_indicator,l]            
        
        Hmtx = coo_matrix((nnz_vals,(I,J)),shape=(self.n1*(self.n2+self.L-1),self.n1*self.n2*self.L))

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
        self.Hmtx = Hmtx

        #return Hmtx.tocsr()

    def take_simulated_snapshot(self,X,SNR):

        if not (X.shape[0] == self.n1 and X.shape[1] == self.n2 and X.shape[2] == self.L):
            raise NameError('Invalid shape. X must be a 3 dimensional array of shape (n1,n2,L).')

        if not hasattr(self,'Hmtx'):
            self.construct_system_mtx()
            print('System matrix Hmtx has been successfully created and added to models attributes.')

        n = self.n1*self.n2*self.L
        m = self.n1*(self.n2 + self.L - 1)
        y = self.Hmtx*np.reshape(X,(n,1),'F')
        
        if SNR != np.inf:
            sigma = np.sqrt(np.std(y,axis=0)**2/(10**(SNR/10)))
            y = y + sigma*np.random.randn(m,1)

        self.Y = np.reshape(y,(self.n1,self.n2 + self.L - 1),'F')
        print('Real coded snapshot Y has been successfully loaded and added to models attributes.')

    def load_real_snapshot(self, Y):
        if not (Y.shape[0] == self.n1 and Y.shape[1] == self.n2 + self.L - 1):
            raise NameError('Invalid shape. Y must be a 2 dimensional array of shape (n1,n2+L-1).')
        if not hasattr(self,'Hmtx'):
            self.construct_system_mtx()
            print("System matrix Hmtx has been successfully created and added to model's attributes.")
        
        self.Y = Y
        print("Real coded snapshot Y has been successfully loaded and added to model's attributes.")

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
        
        omega_ind_tmp = np.unique(self.Hmtx.tocsr()[omega_k_tilde,:].nonzero()[1])
        
        omega_sub_tmp = np.unravel_index(omega_ind_tmp.flatten(), (self.n1,self.n2,self.L), order = 'F')

        z = np.unique(omega_sub_tmp[2])
        l0 = np.min(z)
        L_s = len(z) # number of bands contributing to formation of Y at omega_tilde 

        #x = np.max(omega_sub_tmp[1]) - np.min(omega_sub_tmp[1]) + 1     
        #if (np.max(omega_sub_tmp[1]) - np.min(omega_sub_tmp[1]) + 1) != (width + L_s - 1):
        #    raise Warning(" X at omega may not be a parallelepiped")

        omega_k = {}
        for l in range(l0,L_s): # loop over the bands 

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
        return omega_k_tilde, SignalSubDomain(omega_k)












