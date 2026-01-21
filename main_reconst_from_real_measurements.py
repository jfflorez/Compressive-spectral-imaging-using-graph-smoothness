import os
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix

import matplotlib.pyplot as plt

import sensing_models.sd_cassi as sd_cassi
import utils.gcsi_utils as gcsi_utils
import reconst_algs.gcsi_algs as gcsi_algs
from utils.datasets import load_dataset


datasets_dir = 'datasets'
dataset_name ='real_data_SCN_2_scale_2_June032021_OE.mat'
dataset_path = os.path.join(datasets_dir,dataset_name)

# Load and validate dataset
dataset = load_dataset(dataset_path)

hsdc_name = 'scn2'
print(dataset.keys())


L  = np.max(dataset['lambda_calib'].shape)
n1 = dataset['Y'].shape[0]
n2 = dataset['Y'].shape[1] - L + 1

mask = dataset['mask']
disp_dir = dataset['disp_dir']

sdcassi_model = sd_cassi.SingleDisperserCassiModel(n1,n2,L,mask,disp_dir)
sdcassi_model.load_system_mtx()
sdcassi_model.load_real_snapshot(dataset['Y'])

#sdcassi_model.take_simulated_snapshot(dataset['X'],SNR=np.inf)
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

        num_neigs = 33 #100
        z_k, edge_set_k = gcsi_utils.preprocess_side_image_for_graph_learning(dataset['pan_img'],omega_k,n1,n2,L,num_neigs)

        n = len(z_k)
        m = len(omega_k_tilde)
        vals = np.ones(edge_set_k[0].shape)
        A = coo_matrix((vals,(edge_set_k[0],edge_set_k[1])),shape=(n,n))
        A = A.maximum(A.T)
        q = num_neigs # int(num_neigs/2)

        #L_G = sp.sparse.spdiags(np.sum(A, axis=1).squeeze(), 0 , n , n ) - A
        #L_G = L_G/np.sum(np.sum(A, axis=1).squeeze())

        #plt.figure()
        #plt.plot(L_G@z_k)
        
        try:
            W_G = gcsi_utils.load_graph_on_omega_x0y0(hsdc_name, x0,y0)
        except:
            W_G, theta = gcsi_utils.construct_graph_on_omega_x0y0(z_k,edge_set_k, q)
            #W_G = W_G/W_G[np.unravel_index(W_G.argmax(),W_G.shape)]        
            gcsi_utils.save_graph_on_omega_x0y0(hsdc_name, W_G, x0,y0)

        degrees = np.sum(W_G,axis=1)
        idx_max_degree = np.argmax(degrees.squeeze())

        multi_idx = np.unravel_index(omega_k.to_array(),(n1,n2,L), order = 'F')

        multi_idx_max_degree = np.unravel_index(omega_k.to_array()[idx_max_degree],(n1,n2,L), order = 'F')

        Hmtx = sdcassi_model.Hmtx.tocsr()[omega_k_tilde,:]
        Hmtx = Hmtx[:,omega_k.to_array()]
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
        x_hat, metadata = gcsi_algs.gsm_noisy_case_estimation(Hmtx.tocsr(), y, (W_G).tocsr(), params = {'alpha' : 1.93, 'tol':1e-5, 'maxiter': 10000})        

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

#X_hat = np.maximum(X_hat,0)
    #W[W<1e-5] = 0
    

fig, axes = plt.subplots(1, 3, figsize=(15, 5), width_ratios=[1, 1, 2], constrained_layout=True)


# ---- Figura 1 ----
im0 = axes[0].imshow(
    Y[np.unravel_index(omega_k_tilde, (n1, n2+L-1), 'F')]
    .reshape((height, width), order='F')
)
axes[0].set_title("Original : Y[omega_k_tilde]")
axes[0].set_aspect('equal')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# ---- Figura 2 ----
H = sdcassi_model.Hmtx.tocsr()[omega_k_tilde, :]
H = H[:, omega_k.to_array()]

im1 = axes[1].imshow(
    (H @ X_hat.ravel(order='F')[omega_k.to_array()]).reshape((height, width), order='F')
)
axes[1].set_title("Reconst. : Y_hat[omega_k_tilde]")
axes[1].set_aspect('equal')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# ---- Figura 3 ----
im2 = axes[2].imshow(
    np.reshape(Y, (n1, n2+L-1), order='F')
)
axes[2].set_title("Original measurement snapshot")
axes[2].set_aspect('equal')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)



if not os.path.isdir('figures'):
    os.mkdir('figures')
fig.savefig(f"figures/result_visualizations_for_{dataset_name.replace('.mat','.svg')}")

plt.show()