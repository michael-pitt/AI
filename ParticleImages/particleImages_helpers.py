import numpy as np
import matplotlib.pyplot as plt
import torch


def printProgressBar (iteration, total, loss = '', decimals = 1, length = 50):
    total = total - 1 #since usually we start from 0 till n-1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '#' * filledLength + '-' * (length - filledLength)
    if(loss): print('\rprogress |%s| %s%% loss - %s' % (bar, percent, str(loss)), end = '\r')
    else: print('\rprogress |%s| %s%%' % (bar, percent), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def CellToMatrix(tree, nLayers = 6, nPixEta = 64, nPixPhi = 64, calorSizeXY = 144 * 16):
    
    cs_e,cs_x,cs_y,cs_dx,cs_dy,cs_l,cs_chfrac = tree.arrays(["cell_e",
                                                         "cell_x",
                                                         "cell_y",
                                                         "cell_dx",
                                                         "cell_dy",
                                                         "cell_l",
                                                         "cell_chfrac"]).values()
    nevents = len(cs_e)
    print('using total '+str(nevents)+' events')

    cell_dEta = calorSizeXY // nPixEta;
    cell_dPhi = calorSizeXY // nPixPhi;
    
    Ch_Energy = np.zeros((nevents,nLayers,nPixPhi,nPixEta))
    Nu_Energy = np.zeros((nevents,nLayers,nPixPhi,nPixEta))

    for i, (c_e, c_x, c_y, c_dx, c_dy, c_l, c_chfrac) in enumerate(zip(cs_e,cs_x,cs_y,cs_dx,cs_dy,cs_l,cs_chfrac)):
        printProgressBar(i, nevents)
        for e,x,y,dx,dy,l,f in zip(c_e, c_x, c_y, c_dx, c_dy, c_l, c_chfrac):
            BinEtaMin = int(x-dx/2 + calorSizeXY/2) // cell_dEta
            BinEtaMax = int(x+dx/2 + calorSizeXY/2) // cell_dEta
            BinPhiMin = int(y-dy/2 + calorSizeXY/2) // cell_dPhi
            BinPhiMax = int(y+dy/2 + calorSizeXY/2) // cell_dPhi
            if(BinEtaMax==BinEtaMin): BinEtaMax=BinEtaMax+1
            if(BinPhiMax==BinPhiMin): BinPhiMax=BinPhiMax+1
            denom = (BinEtaMax-BinEtaMin)*(BinPhiMax-BinPhiMin) # democratic
            for bin_eta in range(BinEtaMin,BinEtaMax):
                for bin_phi in range(BinPhiMin,BinPhiMax):
                    Ch_Energy[i][l-1][bin_phi][bin_eta] += e*f/denom
                    Nu_Energy[i][l-1][bin_phi][bin_eta] += e*(1.0-f)/denom
                    
    return Ch_Energy, Nu_Energy
	
def DrawEvent(event, cellNu_Energy, cellCh_Energy):

    fig, ax = plt.subplots(6, 3, figsize=(30, 60))
    Title = ['Input Image', 'Charged Image', 'Neutral Image']
    LayerNames=['ECAL1','ECAL2','ECAL3','HCAL1','HCAL2','HCAL3']

    for layer_i in range(6):

        input_image = cellNu_Energy[event][layer_i] + cellCh_Energy[event][layer_i]
        target_nu_image = cellNu_Energy[event][layer_i]
        target_ch_image = cellCh_Energy[event][layer_i]

        ax[layer_i][0].imshow( input_image, cmap='plasma', vmin=0.001, vmax=10.8 )
        ax[layer_i][1].imshow( target_ch_image, cmap='OrRd', vmin=0.001, vmax=10.8 )
        ax[layer_i][2].imshow( target_nu_image, cmap='Greens', vmin=0.001, vmax=10.8 )

        major_ticks = np.arange(-0.5, 63.5, 1)
        minor_ticks = np.arange(-0.5, 63.5, 1)

        ax[layer_i][0].set_ylabel(LayerNames[layer_i],fontsize=64)
        for ipad in range(3) : 
            ax[layer_i][ipad].set_xticks(minor_ticks, minor=True)
            ax[layer_i][ipad].set_yticks(minor_ticks, minor=True)
            ax[layer_i][ipad].grid(which='minor')
            ax[0][ipad].set_title(Title[ipad]+' ' , fontsize=48)     
        ax[layer_i][0].text(5,5,'E = %2.2f GeV'%(input_image.sum()/1e3),fontsize=64,bbox={'facecolor': 'white'})
        ax[layer_i][1].text(5,5,'E = %2.2f GeV'%(target_ch_image.sum()/1e3),fontsize=64,bbox={'facecolor': 'white'})
        ax[layer_i][2].text(5,5,'E = %2.2f GeV'%(target_nu_image.sum()/1e3),fontsize=64,bbox={'facecolor': 'white'})

    plt.tight_layout()
    fig.savefig('images/Event_' + str(event) + '_cells.pdf')
    fig.savefig('images/Event_' + str(event) + '_cells.png')
    plt.show()
    
def DrawEventSR(data_generator, event_number, model = None, layer=None, device = torch.device("cpu")):

    imageLR, imageHR = data_generator[event_number]
    
    nimage = 2
    out_string = 'images/SR_'
    Title = ['LR image', 'HR image',]
    if model:
        yhat = model([torch.tensor(image[None,:,:,:]) for image in imageLR])
        imageHRbar = yhat.squeeze().to(device).cpu().detach().numpy()
        if layer: imageHRbar = imageHRbar[None,:,:]
        nimage = 3
        Title.append('SR image')
        out_string += 'yhat_'
    Nlayers = 1 if layer else 6
    fig, ax = plt.subplots(Nlayers, nimage, figsize=(5*nimage, 5 if layer else 30))
    LayerNames=['ECAL1','ECAL2','ECAL3','HCAL1','HCAL2','HCAL3']

    list_layers = range(6)
    if layer: list_layers = [layer]
    for layer_i in list_layers:

        LR_image = imageLR[layer_i].squeeze()
        HR_image = imageHR.squeeze()[layer_i]
        if layer: layer_i = 0; axi=ax
        else: axi=ax[layer_i]
            
        axi[0].imshow( LR_image, cmap='plasma', vmin=0.001, vmax=10.8, aspect='auto' )
        axi[1].imshow( HR_image, cmap='plasma', vmin=0.001, vmax=10.8, aspect='auto' )

        xticksLR = np.arange(0.5, LR_image.shape[1]-0.5, 1)
        yticksLR = np.arange(-0.5, LR_image.shape[0]-0.5, 1)
        ticksHR = np.arange(-0.5, HR_image.shape[0]-0.5, 1)
        xticks = [xticksLR, ticksHR, ticksHR]
        yticks = [yticksLR, ticksHR, ticksHR]

        axi[0].set_ylabel(LayerNames[layer_i],fontsize=26)
        for ipad in range(nimage) : 
            axi[ipad].set_xticks(xticks[ipad], minor=True)
            axi[ipad].set_yticks(yticks[ipad], minor=True)
            axi[ipad].grid(which='minor')
            if layer_i==0: axi[ipad].set_title(Title[ipad]+' ' , fontsize=20)     
        axi[0].text(LR_image.shape[1]*0.08,LR_image.shape[0]*0.08,'E = %2.2f GeV'%(LR_image.sum()/1e3),fontsize=26,bbox={'facecolor': 'white'})
        axi[1].text(5.12,5.12,'E = %2.2f GeV'%(HR_image.sum()/1e3),fontsize=26,bbox={'facecolor': 'white'})
        
        if model:
            SR_image = imageHRbar[layer_i]
            axi[2].imshow( SR_image, cmap='plasma', vmin=0.001, vmax=10.8 )
            axi[2].text(5.12,5.12,'E = %2.2f GeV'%(SR_image.sum()/1e3),fontsize=22,bbox={'facecolor': 'white'})

    plt.tight_layout()
    fig.savefig(out_string+'ev_' + str(event_number) + '.pdf')
    fig.savefig(out_string+'ev_' + str(event_number) + '.png')
    plt.show()
	
