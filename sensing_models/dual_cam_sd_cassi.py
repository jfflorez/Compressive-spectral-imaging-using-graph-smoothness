import numpy as np

import scipy
from scipy.sparse import kron
from scipy.sparse import diags
from scipy.sparse import vstack
import matplotlib.pyplot as plt

from sd_cassi import SingleDisperserCassiModel


class DualCameraSDCassiModel:
    """ creates a dual camera single disperser cassi model, where """
    def __init__(self,n1,n2,L,mask,disp_dir,spectral_sen) -> None:
        """
            Parameters:
            sdcassi_model: sd_cassi.SingleDisperserCassiModel object, 

            ss: numpy.ndarray, two dimensional array of shape (K,L), where each row contains the spectral sensitivity of 
            side information camera's channels. 
            For a panchromatic side camera, K=1, and for an RGB camera, K=3.""" 
        self.sdcassi_obj = SingleDisperserCassiModel(n1,n2,L,mask,disp_dir)
        self.n1 = n1
        self.n2 = n2
        self.L = L
        self.K = spectral_sen.shape[0]
        self.spectral_sen = spectral_sen

        if not hasattr(self.sdcassi_obj,'Hmtx'):
            self.sdcassi_obj.construct_system_mtx()
            #raise NameError("Class attribute Hmtx is not defined, i.e., hasattr(sdcassi_obj,'Hmtx') ->  False.")
        #else:
        #self.Hmtx = self.sdcassi_obj.Hmtx
        

        #if not isinstance(sdcassi_obj, sd_cassi.SingleDisperserCassiModel):
        #    raise NameError("Invalid camera model. sdcassi_obj must be an instance of the class SingleDisperserCassiModel")

        if not spectral_sen.shape[1] == self.L:
            raise NameError('Inconsistent shape. spectral_sen must be of shape (K, sdcassi_obj.L)')
    def get_CASSI_mtx(self):
        if not hasattr(self.sdcassi_obj,'Hmtx'):
            return None        
        return self.sdcassi_obj.Hmtx

    def construct_side_system_mtx(self):
        n1n2 = self.n1*self.n2
        for k in range(self.spectral_sen.shape[0]):
            if k > 0:
                self.Rmtx = vstack(self.Rmtx, kron(self.spectral_sen[k,:],diags(np.ones((n1n2,1)).squeeze(), 0, shape = (n1n2,n1n2))))

            else:                
                self.Rmtx = kron(self.spectral_sen[k,:], diags(np.ones((n1n2,1)).squeeze(), 0, shape = (n1n2,n1n2)))
    
    def get_side_camera_mtx(self):
        if not hasattr(self,'Rmtx'):
            return None
        return self.Rmtx
    
    def take_simulated_snapshots(self,X,SNR1,SNR2):

        if not check_shape(X, self.n1, self.n2, self.L):
            raise NameError('Invalid shape. X must be a 3 dimensional array of shape (n1,n2,L).')

        if not hasattr(self.sdcassi_obj,'Hmtx'):
            self.sdcassi_obj.construct_system_mtx()
            print('CASSI System matrix Hmtx has been successfully created and added to models attributes.')

        if not hasattr(self,'Rmtx'):
            self.construct_side_system_mtx()
            print('Side Camera System matrix Rmtx has been successfully created and added to models attributes.')

        n = self.n1*self.n2*self.L
        m = self.n1*(self.n2 + self.L - 1)        
        y = self.sdcassi_obj.Hmtx*np.reshape(X,(n,1),'F')
        z = self.Rmtx*np.reshape(X,(n,1),'F')
        
        if SNR1 != np.inf:
            sigma = np.sqrt(np.std(y,axis=0)**2/(10**(SNR1/10)))
            y = y + sigma*np.random.randn(m,1)

        self.Y = np.reshape(y,(self.n1,self.n2 + self.L - 1),'F')
        print('Real coded snapshot Y has been successfully loaded and added to models attributes.')

        if SNR2 != np.inf:
            sigma = np.sqrt(np.std(z,axis=0)**2/(10**(SNR2/10)))
            z = z + sigma*np.random.randn(m,1)

        self.Y = np.reshape(y,(self.n1,self.n2 + self.L - 1),'F')
        self.Z = np.reshape(z,(self.n1,self.n2,self.K),'F')
        print('Side information snapshot Z has been successfully loaded and added to models attributes.')

    def load_real_snapshots(self, Y, Z):
        if not check_shape(Y, self.n1, self.n2 + self.L - 1, 1):
            raise NameError('Invalid shape. Y must be a 2 dimensional array of shape (n1,n2+L-1).')
        if (not hasattr(self,'Hmtx')) or (not hasattr(self,'Rmty')):
            self.construct_system_mtx()
            print("System matrices Hmtx and Rmtx have been successfully created and added to model's attributes.")

        if not check_shape(Z, self.n1, self.n2, self.K):
            raise NameError('Invalid shape. Z must be a either of shape (n1,n2) or (n1,n2,K).')
        
        self.Y = Y
        self.Z = Z
        print("Real coded snapshot Y has been successfully loaded and added to model's attributes.")

def check_shape(A: np.ndarray, height: int, width: int, depth:int) -> bool:
        if A.ndim == 3:
            return (A.shape[0] == height) and (A.shape[1] == width) and (A.shape[2] == depth)
        elif A.ndim == 2:
            return (A.shape[0] == height) and (A.shape[1] == width)
        else:
            return False

if __name__ == '__main__':
    dataset_dir = 'Datasets/'
    dataset = scipy.io.loadmat(dataset_dir + 'simulated_data_HSDC1_DB_Oct092019_5_OE.mat')
    #dataset = scipy.io.loadmat(dataset_dir + 'real_data_SCN_2_scale_2_June032021_OE')
    print(dataset.keys())


    L  = np.max(dataset['lambda_calib'].shape)
    n1 = dataset['Y'].shape[0]
    n2 = dataset['Y'].shape[1] - L + 1

    if 'X' in dataset:
        x = np.reshape(dataset['X'],(n1*n2*L,1),'F')
    else:
        x = np.zeros((n1,n2,L)); 
        x[:,:,L-1] = 1; 
        x = np.reshape(x,(n1*n2*L,1),'F')

    mask = dataset['mask']
    disp_dir = 'left2right'
    #disp_dir = 'right2left'
    #sdcassi_model = sd_cassi.SingleDisperserCassiModel(n1,n2,L,mask,disp_dir)
    #sdcassi_model.construct_system_mtx()

    dc_sdcassi_model = DualCameraSDCassiModel(n1,n2,L,mask,disp_dir,np.ones((1,L)))


    dc_sdcassi_model.take_simulated_snapshots(dataset['X'],SNR1=20, SNR2=np.inf)
    #Y = sdcassi_model.Y
    #Hmtx = sdcassi_cam.construct_cassi_mtx()

    plt.imshow(dc_sdcassi_model.Y)
    plt.figure(), plt.imshow(dc_sdcassi_model.Z)
    plt.show()
