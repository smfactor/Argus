import numpy as np

def get8(im,i,j):
    if i==255 and j==255:
        a = np.array([im[i-1,j-1],im[i-1,j],im[i,j-1]])
    elif i==255 and j==0:
        a = np.array([im[i-1,j+1],im[i-1,j],im[i,j+1]])
    elif i==0 and j==255:
        a = np.array([im[i+1,j-1],im[i+1,j],im[i,j-1]])
    elif i==255:
        a = np.array([im[i-1,j-1],im[i-1,j],im[i-1,j+1],im[i,j+1],im[i,j-1]])
    elif j==255:
        a = np.array([im[i-1,j-1],im[i-1,j],im[i,j-1],im[i+1,j-1],im[i+1,j]])
    elif i==0 and j==0:
        a = np.array([im[i,j+1],im[i+1,j],im[i+1,j+1]])
    elif i==0:
        a = np.array([im[i,j+1],im[i,j-1],im[i+1,j-1],im[i+1,j],im[i+1,j+1]])
    elif j==0:
        a = np.array([im[i-1,j],im[i-1,j+1],im[i,j+1],im[i+1,j],im[i+1,j+1]])
    else:
        a = np.array([im[i-1,j-1],im[i-1,j],im[i-1,j+1],im[i,j+1],im[i,j-1],im[i+1,j-1],im[i+1,j],im[i+1,j+1]])
    return a

def fixim(im,mask):
    #replace bad pix with nan
    imnan=np.copy(im)
    imnan[np.where(mask)]=np.nan
    fixedim = np.copy(imnan)
    while np.isnan(fixedim).any():
        i,j=np.where(np.isnan(fixedim))
        for n in range(np.size(i)):
            samp=get8(imnan,i[n],j[n])
            fixedim[i[n],j[n]]=np.nanmedian(samp)
        imnan=np.copy(fixedim)
    return fixedim

def findStar(im, xg, yg,r):
    '''
    Finds the flux weighted centroid of im in a circle centered at (xg, yg)
    with radius r.

    :param im:
    image array. convention is im[y,x].

    :param (xg, yg):
    initial guess for the center of the flux peak.

    :param r:
    radius to find the flux weighted centroid within (around (xg, yg)).
    '''
    ysize,xsize=np.shape(im)
    x, y, tf = 0, 0, 0
    #sum flux and flux weighted coordinates
    for i in range(-r,r+1):
        for j in range(-r,r+1):
            if (np.sqrt((i)**2+(j)**2)<=r):
                ygg = (yg+j)%ysize
                xgg = (xg+i)%xsize
                f = im[ygg,xgg]
                tf += f
                x += f*i
                y += f*j

    #normalize by total flux
    x = x/tf
    y = y/tf
    return (x+xg, y+yg)

def readMNAout(path,params,chi2):
    '''
    Reads output of analyzeMN, appends median and 1sigma errorbar values to params array.
    returns array of log likelihood, chi2, and median and 1sigma parameter values

    :param path:
    path to output file

    :param params:
    array to append median and 1sigma parameter values to

    :param chi2:
    chi^2 value of model from kerphCor (also gets appended onto params)
    '''
    f2=open(path,'r')
    #f3=open(path+'chi2','r')
    #skip intro lines
    #garbage=f2.readline()
    garbage=f2.readline()
    garbage=f2.readline()
    sarray=f2.readline().split()
    #garbage=f3.readline()
    #lgZ,slgZ
    params=np.append(params,float(sarray[2]))
    params=np.append(params,float(sarray[4]))
    #chi^2
    params=np.append(params,chi2) #float(f3.readline()))
    garbage=f2.readline()
    #other params
    for line in f2:
        sarray=line.split()
        params=np.append(params,float(sarray[1]))
        params=np.append(params,float(sarray[3]))
    f2.close()
    #f3.close()
    return params

def bpix_close(bad_pix_map, xc, yc):
    '''
    returns the closest bad pixel to the flux centroid

    :param bad_pix_map:
    bad pixel map: non zero values are bad pixels

    :param (xc,yc):
    flux centroid from output of findStar
    '''
    ind=np.where(bad_pix_map)
    ind_yshift=ind[0]-yc
    ind_xshift=ind[1]-xc
    r = np.sqrt(ind_xshift*ind_xshift+ind_yshift*ind_yshift)
    return np.min(r)
