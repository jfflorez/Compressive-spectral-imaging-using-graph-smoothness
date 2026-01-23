from inspect import trace
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt

import sensing_models.sd_cassi as sd_cassi
import utils.gcsi_utils as gcsi_utils
import reconst_algs.gcsi_algs as gcsi_algs
from graphs.structure_learning import build_kalofolias_graph_adj_mtrx


from skimage.transform import rescale
from utils.datasets import load_dataset


dataset_dir = 'datasets'
dataset_filename = 'simulated_data_HSDC1_DB_Oct092019_5_OE.mat'
dataset_filepath = os.path.join(dataset_dir,dataset_filename)
dataset = load_dataset(dataset_filepath)

hsdc_name = 'hsdc1_256by256'
print(dataset.keys())



dataset['X'] = rescale(dataset['X'][:,:,0:26], (0.5,0.5,1), anti_aliasing=True)
dataset['pan_img'] = rescale(dataset['pan_img'], 0.5, anti_aliasing=True)

L  = dataset['X'].shape[2]
n1 = dataset['X'].shape[0]
n2 = dataset['X'] .shape[1]


if 'X' in dataset:
    x = np.reshape(dataset['X'],(n1*n2*L,1),'F')
else:
    x = np.zeros((n1,n2,L)); 
    x[:,:,L-1] = 1; 
    x = np.reshape(x,(n1*n2*L,1),'F')

t = 0.5
# Borrow mask from real data, which has compatible dimensions with downsampled X
dataset1 = load_dataset(os.path.join(dataset_dir,'real_data_SCN_2_scale_2_June032021_OE.mat'))
mask = dataset1['mask']
#mask = gcsi_utils.generate_sd_cassi_calibration_cube(t,n1,n2,L)

#disp_dir = 'left2right'
disp_dir = dataset1['disp_dir']
sdcassi_model = sd_cassi.SingleDisperserCassiModel(n1,n2,L,mask,disp_dir)
sdcassi_model.load_system_mtx()
#sdcassi_model.load_real_snapshot(dataset['Y'])

sdcassi_model.take_simulated_snapshot(dataset['X'],SNR=np.inf)
Y = sdcassi_model.Y
#Hmtx = sdcassi_cam.construct_cassi_mtx()

height= 32
width = 32
#xx=np.linspace(0,n2+L-1 - 1 - (width-1), int(2*(n2+L-1)/width), dtype=int)
#yy=np.linspace(0,n1 - 1 - (height-1), int(2*n1/height), dtype=int)
#xx=np.linspace(0,n2+L-1 - 1 - (width-1), int((n2+L-1)/width), dtype=int)
#yy=np.linspace(0,n1 - 1 - (height-1), int(n1/height), dtype=int)
xx = np.arange(0,n2+L-1, width, dtype= int)
yy = np.arange(0,n1, height, dtype= int)

X_hat = np.zeros((n1,n2,L))
C = np.zeros((n1,n2,L))


for j in range(len(xx)):
    for i in range(len(yy)):
        x0 = xx[j]
        y0 = yy[i]
        height = min(y0+height,n1)-y0
        width = min(x0+width,n2+L-1)-x0

        omega_k_tilde, omega_k = sdcassi_model.get_system_submtx_pair(k = (x0,y0,height,width))
        print('x0:',x0,'L_s: ', str(omega_k.L_),'size:',omega_k.to_array().size)

        side_img = dataset['pan_img']
        spectral_img_shape = (n1,n2,L)
        num_neigs = 33 #100
        
        #try:
        #    W_G = gcsi_utils.load_graph_on_omega_x0y0(hsdc_name, x0,y0)
        #except:
        #    W_G, theta = gcsi_utils.construct_graph_on_omega_x0y0(z_k,edge_set_k, q=num_neigs)  
            #gcsi_utils.save_graph_on_omega_x0y0(hsdc_name, W_G, x0,y0)
        #W_G, theta = gcsi_utils.construct_graph_on_omega_x0y0(z_k,edge_set_k, q=num_neigs) 


        W_G = build_kalofolias_graph_adj_mtrx(side_img, omega_k, spectral_img_shape, num_neigs) 

            
        degrees = np.sum(W_G,axis=1)
        idx_max_degree = np.argmax(degrees.squeeze())

        multi_idx = np.unravel_index(omega_k.to_array(),(n1,n2,L), order = 'F')

        multi_idx_max_degree = np.unravel_index(omega_k.to_array()[idx_max_degree],(n1,n2,L), order = 'F')

        Hmtx = sdcassi_model.Hmtx.tocsr()[omega_k_tilde,:]
        Hmtx = Hmtx[:,omega_k.to_array()]
        m, n = Hmtx.shape
        y = Y[np.unravel_index(omega_k_tilde,(n1,n2+L-1), order = 'F')].reshape((m,1),order = 'F')  

        """     plt.figure()
                ax0 = plt.gca()
                img = ax0.imshow(Y)
                plt.colorbar(img,ax=ax0)

                plt.figure()
                ax0 = plt.gca()
                img = ax0.imshow(y.reshape((height,width),order='F'))
                plt.colorbar(img,ax=ax0)"""

        
        #W_ = coo_matrix((np.ones(e[0].shape),(e[0],e[1])), shape = (n,n))
        #W_ = W_.maximum(W_)

        x_hat, solver_info = gcsi_algs.gsm_noiseless_case_estimation(Hmtx.tocsr(), y, (W_G).tocsr(), params = {'alpha' : 10,'tol':1e-6, 'maxiter': 10000})        

        #multi_idx = np.unravel_index(omega_k.to_array(),(n1,n2,L), order = 'F')

        plot_flag = False

        if plot_flag:
#            x_ask = dataset['X'][multi_idx]

            fig, ax = plt.subplots(1,2)
            width_ = np.max(multi_idx[1][multi_idx[2]==16])-  np.min(multi_idx[1][multi_idx[2]==16]) + 1
            height_ = np.max(multi_idx[0][multi_idx[2]==16])-  np.min(multi_idx[0][multi_idx[2]==16]) + 1

            z_l = z_k[multi_idx[2]==16]
            x_l_hat = x_hat[multi_idx[2]==16]
            #x_l_ask = x_ask[multi_idx[2]==16]
            

            ax[0].imshow(np.reshape(z_l , (height_,width_), order = 'F'))    
            ax[1].imshow(np.reshape(x_l_hat , (height_,width_), order = 'F'))  
            #ax[2].imshow(np.reshape(x_l_ask, (height_,width_), order = 'F'))  
            plt.show()
        
        X_hat[multi_idx] += x_hat
        C[multi_idx] += 1 

X_hat = X_hat/C

# Clip reconstruction to non-negative values
X_hat_clipped = np.maximum(X_hat, 0)

# Compute PSNR
psnr_val = gcsi_utils.psnr(dataset['X'], X_hat_clipped)
print(f"PSNR: {psnr_val[0]:.2f} dB")

# Create figure
fig, axes = plt.subplots(
    1, 4,
    figsize=(15, 5),
    constrained_layout=True
)

# Select evenly spaced wavelength indices
num_bands = 4
wavelength_idx = np.linspace(0, L - 1, num_bands, dtype=int)

for ax, idx in zip(axes, wavelength_idx):
    error_img = X_hat[:,:,idx] - dataset['X'][:, :, idx]

    im = ax.imshow(error_img, cmap="bwr")
    lam = dataset['lambda_calib'].flatten()[idx]
    ax.set_title(rf"Reconstruction error at $\lambda$ = {lam}")
    ax.axis("off")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Ensure output directory exists
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Save figure
dataset_name = dataset_filename.replace(".mat", "")
fig.savefig(
    os.path.join(output_dir, f"error_images_avgPSNR_{psnr_val[0]:.2f}dB_for_{dataset_name}.svg"),
    dpi=300
)

plt.show()




    
