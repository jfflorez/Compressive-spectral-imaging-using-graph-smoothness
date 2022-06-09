from inspect import trace
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sensing_models.sd_cassi as sd_cassi
import utils.gcsi_utils as gcsi_utils
import reconst_algs.gcsi_algs as gcsi_algs

from scipy.sparse import coo_matrix

from skimage.transform import rescale



dataset_dir = 'C:\\Users\\juanf\\OneDrive\\Documents\\GitHub\\Compressive-spectral-imaging-using-graph-smoothness\\datasets\\'
dataset = sp.io.loadmat(dataset_dir + 'simulated_data_HSDC1_DB_Oct092019_5_OE.mat')
#dataset = sp.io.loadmat(dataset_dir + 'real_data_SCN_2_scale_2_June032021_OE')

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
dataset1 = sp.io.loadmat(dataset_dir + 'real_data_SCN_2_scale_2_June032021_OE')
mask = dataset1['mask']
#mask = gcsi_utils.generate_sd_cassi_calibration_cube(t,n1,n2,L)

#disp_dir = 'left2right'
disp_dir = 'right2left'
sdcassi_model = sd_cassi.SingleDisperserCassiModel(n1,n2,L,mask,disp_dir)
sdcassi_model.construct_system_mtx()
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
#xx = [86]
#yy = [90]

X_hat = np.zeros((n1,n2,L))
C = np.zeros((n1,n2,L))

"""for j in range(len(xx)):
    for i in range(len(yy)):
        x0 = xx[j]
        y0 = yy[i]

        omega_k_tilde, omega_k = sdcassi_model.get_system_submtx_pair(k = (x0,y0,height,width))
        print('x0:',x0,'L_s: ', str(omega_k.L_),'size:',omega_k.to_array().size)

        multi_idx = np.unravel_index(omega_k.to_array(),(n1,n2,L), order = 'F')
        X_hat[multi_idx] += dataset['X'][multi_idx]
        C[multi_idx] += 1

X_hat = X_hat/C"""


for j in range(len(xx)):
    for i in range(len(yy)):
        x0 = xx[j]
        y0 = yy[i]

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

        """        node_idx = {}
                for ii in range(len(omega_k.to_array())):
                    node_idx[omega_k.to_array()[ii]] = ii

                x_min = multi_idx[1].min()
                x_max = multi_idx[1].max()

                v_0 = multi_idx_max_degree
                v_nxt = multi_idx_max_degree
                cntr = 0
                while v_nxt[1] < x_max-1 and v_nxt[2] < multi_idx[2].max()-1:       
                    v_nxt = (v_0[0],v_0[1]+1,v_0[2]+1)

                    if cntr > 0 :
                        e_tmp = np.zeros((2,1))
                        e_tmp[0] = node_idx[np.ravel_multi_index(v_0,(n1,n2,L),order='F')]
                        e_tmp[1] = node_idx[np.ravel_multi_index(v_nxt,(n1,n2,L),order='F')]
                        e = np.hstack((e,e_tmp))                
                    else:
                        e = np.zeros((2,1))
                        e[0] = node_idx[np.ravel_multi_index(v_0,(n1,n2,L),order='F')]
                        e[1] = node_idx[np.ravel_multi_index(v_nxt,(n1,n2,L),order='F')]
                    cntr += 1
                    v_0 = v_nxt """

        


        #W_G[W_G.nonzero()] = np.multiply(W_G[W_G.nonzero()], (W_G[W_G.nonzero()]>1e-4).astype(np.float64))

        #L_G = sp.sparse.spdiags(np.sum(W_G, axis=1).squeeze(), 0 , n , n ) - W_G
        #L_G = L_G/np.sum(np.sum(W_G, axis=1).squeeze())
        
        #plt.figure()
        #plt.plot(L_G@z_k)

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

        x_hat = gcsi_algs.gsm_noiseless_case_estimation(Hmtx.tocsr(), y, (W_G).tocsr(), params = {'tol':1e-5, 'maxiter': 10000})        

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

psnr_val = gcsi_utils.psnr(dataset['X'],np.maximum(X_hat,0))

X_hat = np.maximum(X_hat,0)
    #W[W<1e-5] = 0
    
