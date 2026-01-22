import sys
import os

try:
    thisFilePath = os.path.abspath(__file__)
except NameError:
    print("Error: __file__ is not available. 'thisFilePath' will resolved to os.getcwd().")
    thisFilePath = os.getcwd()  # Use current directory or specify a default

projectPath = os.path.normpath(os.path.join(thisFilePath,'..','..'))

if not projectPath in sys.path:
    sys.path.append(projectPath)

import numpy as np
import scipy as sp
from scipy.sparse import kron
from scipy.sparse import diags
from scipy.sparse import vstack
import matplotlib.pyplot as plt

from sensing_models.sd_cassi import SingleDisperserCassiModel, construct_system_mtx
from skimage.transform import rescale
import utils.gcsi_utils as gcsi_utils
from utils.datasets import load_dataset

class DualCameraSDCassiModel:
    """Creates a dual camera single disperser CASSI model from a suitable dataset in .mat format."""
    
    def __init__(self, dataset) -> None:
        """
        Parameters 
        ----------
        dataset: str or dict
            Valid path to a .mat file or pre-loaded dictionary containing a dual camera SD CASSI dataset.
        """
        # Load and validate dataset
        validated_dataset = load_dataset(dataset)
        self.dataset_name = os.path.basename(dataset) if isinstance(dataset, str) else None

        # Extract dimensions
        n1, m2, L = validated_dataset['mask'].shape
        n2 = m2 - L + 1

        # Initialize CASSI model
        self.sdcassi_obj = SingleDisperserCassiModel(
            n1, n2, L, 
            validated_dataset['mask'], 
            validated_dataset['disp_dir']
        )

        # Store attributes
        self.n1 = n1
        self.n2 = n2
        self.L = L
        self.spectral_sen = validated_dataset['spectral_sen'] if 'spectral_sen' in validated_dataset.keys() else np.ones((1,L))
        self.K = self.spectral_sen.shape[0]

    def prepare_for_pickle(self):
        """Call this before pickling to remove large matrices."""
        if hasattr(self.sdcassi_obj, 'Hmtx'):
            delattr(self.sdcassi_obj, 'Hmtx')
        if hasattr(self, '_Rmtx'):
            delattr(self, '_Rmtx')

    @property
    def Hmtx(self):
        """Lazy-loaded CASSI system matrix. Builds on first access and caches for subsequent calls."""
        if not hasattr(self.sdcassi_obj, 'Hmtx'):
            self.sdcassi_obj.load_system_mtx()
        return self.sdcassi_obj.Hmtx

    @property
    def Rmtx(self):
        """Lazy-loaded side camera system matrix. Builds on first access and caches for subsequent calls."""
        if not hasattr(self, '_Rmtx'):
            self._construct_side_system_mtx()
        return self._Rmtx
    @property
    def Y(self):
        return self.sdcassi_obj.Y

    def _construct_side_system_mtx(self):
        """Internal method to construct side camera system matrix."""
        n1n2 = self.n1 * self.n2
        for k in range(self.spectral_sen.shape[0]):
            if k > 0:
                self._Rmtx = vstack(
                    self._Rmtx, 
                    kron(self.spectral_sen[k,:], diags(np.ones((n1n2,1)).squeeze(), 0, shape=(n1n2,n1n2)))
                )
            else:                
                self._Rmtx = kron(
                    self.spectral_sen[k,:], 
                    diags(np.ones((n1n2,1)).squeeze(), 0, shape=(n1n2,n1n2))
                )

    def get_dataset_name(self):
        if not hasattr(self, 'dataset_name'):
            return None
        return self.dataset_name
    
    def get_CASSI_mtx(self):
        """Returns the CASSI system matrix, building it if necessary."""
        return self.Hmtx

    def construct_side_system_mtx(self):
        """Public method to explicitly construct side camera matrix. Typically not needed due to lazy loading."""
        if not hasattr(self, '_Rmtx'):
            self._construct_side_system_mtx()
            print('Side Camera System matrix Rmtx has been successfully created and added to models attributes.')
    
    def get_side_camera_mtx(self):
        """Returns the side camera system matrix, building it if necessary."""
        return self._Rmtx
    
    def take_simulated_snapshots(self, X, SNR1: float = np.inf, SNR2: float = np.inf):
        if not check_shape(X, self.n1, self.n2, self.L):
            raise NameError('Invalid shape. X must be a 3 dimensional array of shape (n1,n2,L).')

        # Matrices are accessed via properties - they build automatically if needed
        n = self.n1 * self.n2 * self.L
        m = self.n1 * (self.n2 + self.L - 1)        
        
        y = self.Hmtx * np.reshape(X, (n,1), 'F')
        z = self.Rmtx * np.reshape(X, (n,1), 'F')
        
        if SNR1 != np.inf:
            sigma = np.sqrt(np.std(y, axis=0)**2 / (10**(SNR1/10)))
            y = y + sigma * np.random.randn(m, 1)

        self.Y = np.reshape(y, (self.n1, self.n2 + self.L - 1), 'F')
        print('Real coded snapshot Y has been successfully loaded and added to models attributes.')

        if SNR2 != np.inf:
            sigma = np.sqrt(np.std(z, axis=0)**2 / (10**(SNR2/10)))
            z = z + sigma * np.random.randn(self.n1 * self.n2 * self.K, 1)

        self.Z = np.reshape(z, (self.n1, self.n2, self.K), 'F')
        print('Side information snapshot Z has been successfully loaded and added to models attributes.')

    def load_real_snapshots(self, Y, Z):
        if not check_shape(Y, self.n1, self.n2 + self.L - 1, 1):
            raise NameError('Invalid shape. Y must be a 2 dimensional array of shape (n1,n2+L-1).')
        if not check_shape(Z, self.n1, self.n2, self.K):
            raise NameError('Invalid shape. Z must be a either of shape (n1,n2) or (n1,n2,K).')
        
        self.sdcassi_obj.load_real_snapshot(Y)
        self.Z = Z
        print("Real coded snapshot Y has been successfully loaded and added to model's attributes.")

    def get_linear_system(self, block: tuple[int, int, int, int] = None):
        return self.sdcassi_obj.get_linear_system(block)

def check_shape(A: np.ndarray, height: int, width: int, depth:int) -> bool:
        if A.ndim == 3:
            return (A.shape[0] == height) and (A.shape[1] == width) and (A.shape[2] == depth)
        elif A.ndim == 2:
            return (A.shape[0] == height) and (A.shape[1] == width)
        else:
            return False
        
def generate_DualCameraSDCassiData(X, lambda_calib, rescaling_factor = 1, t = 0.5, disp_dir = 'left2right'):

    """
    Docstring for generate_DualCameraSDCassiData
    
    Parameters:
    -----------
    X (np.array) : 
        Spectral image of size n1 by n2 by L
    rescaling_factor (float) : 
        Value in (0,1] for downsampling to rescaling_factor*n1 by rescaling_factor*n2 resolution 
    t (float) : 
        Value in (0,1) indicating the transmittance of the coded aperture
    disp_dir (str): 
        Value of either "left2right" or "right2left" indicating the direction in which the coded apereture shadow moves with wavelength
    """


    dataset = {}
    if rescaling_factor < 1:
        dataset['X'] = rescale(X, (rescaling_factor,rescaling_factor,1), anti_aliasing=True)
    else:
        dataset['X'] = X
    n1, n2, L = dataset['X'].shape

    if lambda_calib.size != L:
        raise ValueError(f'Length mismatch. Parameter "lambda_calib" must be a one dimensional array of {L} elements.')
    dataset['lambda_calib'] = lambda_calib
    # Extend this to multi channel side information
    dataset['pan_img'] = np.mean(dataset['X'],axis=2)
    dataset['spectral_sen'] = np.ones((1,L))/L

    dataset['mask'] = gcsi_utils.generate_sd_cassi_calibration_cube(t,n1,n2,L) 

    # Generate CASSI snapshot
    Hmtx = construct_system_mtx(dataset['mask'],n1,n2,L,disp_dir)
    y = Hmtx @ dataset['X'].ravel(order='F') # no need to reshape apparently numpy handles well the multiplication
    dataset['Y'] = np.reshape(y,shape=(n1,n2+L-1))

    return dataset


if __name__ == '__main__':
    

    dataset_dir = 'datasets'
    dataset_filepath = os.path.join(dataset_dir,'simulated_data_HSDC1_DB_Oct092019_5_OE.mat')
    
    # Load existing dual camera sd cassi dataset
    dataset = load_dataset(dataset_filepath)
    print(dataset.keys())
    # Uncommont lines below if you would like to regenerate the dataset with lower image resolution
    # 
    #dataset = generate_DualCameraSDCassiData(dataset['X'],
    #                                         dataset['lambda_calib'],
    #                                         rescaling_factor=0.5,
    #                                         t = 0.5,
    #                                         disp_dir = 'left2right')

    # Compute dimensions of the underlying spectral image based on CASSI snapshot (measurement)
    L  = np.max(dataset['lambda_calib'].shape) # should be same as
    n1 = dataset['Y'].shape[0]
    # The number of columns of the spectral image is given by
    n2 = dataset['Y'].shape[1] - L + 1

    if 'X' in dataset:
        x = np.reshape(dataset['X'],(n1*n2*L,1),'F')
    else:
        x = np.zeros((n1,n2,L)); 
        x[:,:,L-1] = 1; 
        x = np.reshape(x,(n1*n2*L,1),'F')

    # TODO: Estimate disp_dir from dataset['mask']. 
    #mask = dataset['mask']    
    #for l in range(0,L,3):
    #    print(l)
    #    plt.figure(l)
    #    plt.imshow(mask[:,:,l])
    #    plt.title(f'System response at {l}')
    #plt.show()
    # Naive solution: Assume a direction. Construct system matrix and get the response for a 
    # delta spectral images, all ones at wavelenth l and zeroes elsewhere. Then, compare
    # the response with mask at wavelength. They should be the same. Otherwise, the direction is the opposite 
    
    #disp_dir = 'right2left'
    #sdcassi_model = sd_cassi.SingleDisperserCassiModel(n1,n2,L,mask,disp_dir)
    #sdcassi_model.construct_system_mtx()

    dc_sdcassi_model = DualCameraSDCassiModel(dataset) 
    # TODO: If dataset contains snapshots, load them directly. Modify DualCameraModel class.
    dc_sdcassi_model.load_real_snapshots(dataset['Y'],dataset['pan_img'])
    # Uncomment line below to generate new snapshots based on loaded DUAL CAM SD CASSI system.
    #dc_sdcassi_model.take_simulated_snapshots(dataset['X'],SNR1=20, SNR2=np.inf)
    
    plt.imshow(dc_sdcassi_model.Y)
    plt.figure(), plt.imshow(dc_sdcassi_model.Z)
    plt.show()
