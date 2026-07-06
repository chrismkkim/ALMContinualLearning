import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import os


# version = 'version 0.251010.2006'
version = 'version 0.251008.1146'
savepath = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/NuoLi/code/analysis/' + version + '/'
datapath = '/Users/kimchm/Documents/KimNiNature2024/' + version + '/'
# datapath = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/NuoLi/001188/'
animaldirs = [name for name in os.listdir(datapath) 
           if os.path.isdir(os.path.join(datapath, name))]
num_animals = len(animaldirs)

# animal_ix = 0
for animal_ix in range(num_animals):    
    
    animalpath = datapath + animaldirs[animal_ix]

    sess_files = [name for name in os.listdir(animalpath) 
            if os.path.isfile(os.path.join(animalpath, name))]
    num_sess = len(sess_files)

    # sess_file_ix = 0
    for sess_file_ix in range(num_sess):
        
        print('animal ' + str(animal_ix) + ' / ' + str(num_animals) + ', session ' + str(sess_file_ix) + ' / ' + str(num_sess))
        
        animal = animaldirs[animal_ix]
        sess   = sess_files[sess_file_ix]
        sesspath       = animalpath + '/' + sess_files[sess_file_ix]
        save_sess_path = savepath + animal + '/' + sess

        os.makedirs(save_sess_path)

        io = NWBHDF5IO(sesspath, mode="r")
        nwbfile = io.read()

        # file_twophoton = open(savepath + 'two_photon.text','w')

        # '''
        # two photon series
        # '''
        # imgacq = nwbfile.acquisition["TwoPhotonSeries"].data[:]
        # file_twophoton.write(sess + f": {len(imgacq)}\n")

        '''
        image segmentation
        '''
        imgseg = nwbfile.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
        imgseg_vm = imgseg.voxel_mask[:]
        nroi = imgseg_vm.shape[0]
        fov_imgseg0 = np.zeros((512,512))
        for pix in range(nroi):
            x = imgseg_vm[pix][0]
            y = imgseg_vm[pix][1]
            w = imgseg_vm[pix][3]
            fov_imgseg0[y,x] = w
        plt.figure(figsize=(8,8))
        plt.imshow(fov_imgseg0, aspect='auto')    
        plt.tight_layout()
        plt.savefig(save_sess_path + '/imgseg.pdf')


        '''
        roi
        '''
        rois = nwbfile.processing["ophys"]["Fluorescence"]["ROI Response Series"].rois[:]
        # print('number of ROIs: ' + str(rois.shape[0]))
        rois_vm = rois.voxel_mask
        nroi = rois_vm.shape[0]
        wgt = np.array([],dtype=np.float32)
        xcoord = np.array([],dtype=np.float32)
        ycoord = np.array([],dtype=np.float32)
        fov = np.zeros((nroi,512,512))
        for roi in range(nroi):
            roi0 = rois_vm[roi]
            npix = roi0.shape[0]
            for pix in range(npix):
                x = roi0[pix][0]
                y = roi0[pix][1]
                w = roi0[pix][3]
                fov[roi,y,x] = w
                wgt = np.append(wgt,w)
                xcoord = np.append(xcoord,x)
                ycoord = np.append(ycoord,y)
        fov_sum = np.sum(fov,axis=0)    
        plt.figure(figsize=(8,8))
        plt.imshow(fov_sum)    
        plt.tight_layout()
        plt.savefig(save_sess_path + '/_1_roi.pdf')


        data = nwbfile.processing["ophys"]["Fluorescence"]["ROI Response Series"].data[:]
        plt.figure()
        plt.imshow(np.transpose(data), cmap='jet', aspect='auto', vmin=0, vmax=300)
        plt.xlabel('time steps')
        plt.ylabel('neurons')
        plt.tight_layout()
        plt.savefig(save_sess_path + '/_2_roi_response.pdf')

        rndcells = np.random.permutation(nroi)[:25]
        plt.figure(figsize=(15,8))
        for i in range(25):
            plt.subplot(5,5,i+1)
            cell = rndcells[i]
            plt.plot(data[-500:,cell], lw=0.5, label='cell ' + str(cell))
            plt.plot(data[-500:,0], lw=0.5, label='cell 0')
            plt.legend(fontsize=6, loc=1)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(save_sess_path + '/_3_roi_response_comp.pdf')
