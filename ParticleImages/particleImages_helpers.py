import numpy as np
import matplotlib.pyplot as plt


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
    
def DrawEventSR(data_generator, event_number):

    imageLR, imageHR = data_generator[event_number]
    
    fig, ax = plt.subplots(6, 2, figsize=(25, 60))
    Title = ['LR image', 'HR image',]
    LayerNames=['ECAL1','ECAL2','ECAL3','HCAL1','HCAL2','HCAL3']

    for layer_i in range(6):

        LR_image = imageLR[0][layer_i]
        HR_image = imageHR[0][layer_i]

        ax[layer_i][0].imshow( LR_image, cmap='plasma', vmin=0.001, vmax=10.8 )
        ax[layer_i][1].imshow( HR_image, cmap='plasma', vmin=0.001, vmax=10.8 )

        major_ticks = np.arange(-0.5, 63.5, 1)
        minor_ticks = np.arange(-0.5, 63.5, 1)

        ax[layer_i][0].set_ylabel(LayerNames[layer_i],fontsize=64)
        for ipad in range(2) : 
            ax[layer_i][ipad].set_xticks(minor_ticks, minor=True)
            ax[layer_i][ipad].set_yticks(minor_ticks, minor=True)
            ax[layer_i][ipad].grid(which='minor')
            ax[0][ipad].set_title(Title[ipad]+' ' , fontsize=48)     
        ax[layer_i][0].text(5,5,'E = %2.2f GeV'%(LR_image.sum()/1e3),fontsize=64,bbox={'facecolor': 'white'})
        ax[layer_i][1].text(5,5,'E = %2.2f GeV'%(HR_image.sum()/1e3),fontsize=64,bbox={'facecolor': 'white'})

    plt.tight_layout()
    fig.savefig('images/SR_ev_' + str(event_number) + '.pdf')
    fig.savefig('images/SR_ev_' + str(event_number) + '.png')
    plt.show()
	
