import numpy as np
#import scipy.fftpack as fft
import numpy.fft as fft
import scipy.interpolate as interp
import numpy.linalg as linalg

def kerPhaMat(mask,xg,yg):
    '''
    Calculates the kernel phase transfer matrix and related quantities from a given model aperture.
    Returns a tuple of relavent quantities (d,R,A,K,U,V,Wt):
    d: list of baselines sampled by the mask where real(d) adn imag(d) are the x and y coordinates respectively
    R: The 'redundancy' matrix encoding the amound of redundancy in each baseline
    A: The phase transfer matrix encoding the contributions of each aperture to a given measured phase
    K: The kernel-phase transfer matrix
    U, V, Wt: The components of the SVD of A ###### These are probably not needed#####
    
    :param mask:
    2D matrix of 1's and 0's indicating the location of simulated apertures.
    
    :param xg:
    2D matrix of x coordinates indicating the physical location of each grid point.
    
    :param yg:
    2D matrix of y coordinates indicating the physical location of each grid point.
    '''

    #keep track of coordinates of apertures
    apertures=np.where(mask==1)
    xa = xg[apertures]
    ya = yg[apertures]
    spacing=xa[1]-xa[0]
    n=np.size(xa)

    #measure all the baselines
    d=np.zeros((n,n))+1j*np.zeros((n,n))

    for i in range(n):
        #skip lower diagonal half as they are redundant
        d[i][i:]=np.round((xa[i]-xa[i:])/spacing)*spacing+1j*np.round((ya[i]-ya[i:])/spacing)*spacing
    #spacing=np.min(i for i in d if i != 0.)
    #d=np.roud(d/spacing)*spacing

    #get unique baselines
    uds, inv, cou = np.unique(d, return_inverse=1, return_counts=1)

    #R=1/number of redundancies
    #Ri=np.diag(1./cou[np.where(uds!=0+0j)])
    #R=linalg.inv(Ri)
    R=np.diag(cou[np.where(uds!=0+0j)])

    #1 and -1 for each contributing baseline
    m=np.size(uds)
    A=np.zeros((m,n))
    for i in range(m):
        bs = np.where(inv==i)[0]
        b1 = np.divide(bs,n)
        b2 = np.mod(bs,n)
        #A[i,b1] += 1.
        #A[i,b2] -= 1.
        A[i,b1] = 1.
        A[i,b2] = A[i,b2]-1. #incase 2 baselines share an aperture

    #get rid of 0 baseline
    A=A[np.where(uds!=0+0j),:].squeeze()

    #calculate kernel phase matrix
    U,W,Vt = linalg.svd(A,full_matrices=1)
    U=U.squeeze(); W=W.squeeze(); Vt=Vt.squeeze()
    npad = np.shape(U)[1]-np.size(W)
    W=np.pad(W,((0,npad)),'constant')
    K=U[:,np.where(np.isclose(W,0.))].T.squeeze()

    return uds[np.where(uds!=0+0j)],R,A,K,U,W,Vt

def kerph(ph,K,d,R,l,freqx,freqy,err=0):
    '''
    Calculates the kernel-phases from a 'image' of phases.
    Returns a list of kernel-phases.
    
    :param ph:
    'image' of phase information
    
    :param K:
    Kernel-phase transfer matrix.
    
    :param d:
    list of baselines where real(d) and imag(d) are the x and y coordinates respectively.
    
    :param R:
    The 'redundancy' matrix encoding the amound of redundancy in each baseline.
    
    :param l:
    Wavelength of observation in m
    
    :param freqx:
    x frequencies of the points in ph (produced from fftshift(fftfreq))
    
    :param freqy:
    y frequencies of the points in ph (produced from fftshift(fftfreq))
    '''
    nd=len(d)
    uvs = d/l
    fgx, fgy = np.meshgrid(freqx,freqy)

    # 'aperture' size
    df = np.min(np.abs(d))/l/2.
    #phases in apertures    
    pph=np.array([ ph[np.where((fgx-np.real(i))**2+(fgy-np.imag(i))**2<=df*df)] for i in uvs ])
    
    if err:
        #mean and std of phases
        meStd = np.array([[np.mean(i),np.std(i)] for i in pph])
        return np.dot(K,np.dot(R,meStd[:,0])),np.sqrt(np.dot(K*K,np.dot(R*R,meStd[:,1]*meStd[:,1])))
    else:
        phase = np.array([np.mean(i) for i in pph])
        return np.dot(K,np.dot(R,phase))

def kerphA(fph,K,d,R,l):
    '''
    Calculates the kernel-phases from function which returns phases.
    Returns a list of kernel-phases.
    '''
    uvs = d/l
    phase = fph(np.real(uvs),np.imag(uvs))
    return np.dot(K,np.dot(R,phase))
    #return np.dot(K,phase)

def calcPhase(im,ps):
    '''
    Calculates the phase from an image and pixel scale.
    Returns a tuple of (the phase 'image', the frequencies in the x direction, the frequencies in the y direction). 
    
    :param im:
    Image to take the fft of.
    
    :param ps:
    Pixel scale in arc sec/pix.
    '''
    ftim=fft.fftshift(fft.fft2(fft.fftshift(im.astype(float))))
    ph=np.angle(ftim)
    freqx=fft.fftshift(fft.fftfreq(np.shape(im)[1],d=np.radians(ps/3600.)))
    freqy=fft.fftshift(fft.fftfreq(np.shape(im)[0],d=np.radians(ps/3600.)))
    return ph,freqx,freqy

def calcPhaseShift(im,ps,x,y):
    '''
    Calculates the phase from an image and pixel scale and shifts the image in the fourier domain. 
    Returns a tuple of (the phase 'image', the frequencies in the x direction, the frequencies in the y direction). 
    
    :param im:
    Image to take the fft of.
    
    :param ps:
    Pixel scale in arc sec/pix.

    :param x:
    x coordinate to shift to 0.

    :param y:
    y coordinate to shift to 0.
    '''

    ftim=fft.fftshift(fft.fft2(fft.fftshift(im.astype(float))))
    freqx=fft.fftshift(fft.fftfreq(np.shape(im)[1],d=np.radians(ps/3600.)))
    freqy=fft.fftshift(fft.fftfreq(np.shape(im)[0],d=np.radians(ps/3600.)))
    freqxg,freqyg = np.meshgrid(freqx,freqy)

    dy,dx=(np.array(np.shape(im))/2-[y,x])*np.radians(ps/3600.)
    shift = np.exp(2.*np.pi*1.j*(dy*freqyg+dx*freqxg))
    ph = np.angle(ftim/shift)
    return ph,freqx,freqy


def makeGrid(K,d,R,l,name="testGrid", rmin=0.01, rmax=0.5, nr=100, npa=72, fmin=1., fmax=20., nf=100):
    rs=np.logspace(np.log10(rmin),np.log10(rmax),num=nr)
    pas=np.linspace(0.,2.*np.pi,num=npa,endpoint=0)
    fs=np.logspace(np.log10(fmin),np.log10(fmax),num=nf)
    
    kers = np.zeros((nr,npa,nf,np.shape(K)[0]))
    for i in range(nr):
        if i%(nr/10)==0:
            print(i/(nr/10),"% done")
        for j in range(npa):
            for k in range(nf):
                #p=[np.log10(rs[i]),pas[j],np.log10(fs[k])]
                #fmod=modelArpa(p)
                p=[rs[i]*np.cos(pas[j]+np.pi/2.),rs[i]*np.sin(pas[j]+np.pi/2.),np.log10(fs[k])]
                fmod = modelAxy(p)
                kers[i,j,k]=kerphA(fmod,K,d,R,l)
    np.save(name+"_kers.npy",kers)
    np.save(name+"_rs.npy",rs)
    np.save(name+"_pas.npy",pas)
    np.save(name+"_fs.npy",fs)

def gridSearch(kerphdat,sig,gridName="testGrid"):
    rs=np.load(gridName+"_rs.npy")
    pas=np.load(gridName+"_pas.npy")
    fs=np.load(gridName+"_fs.npy")
    nr,npa,nf=len(rs),len(pas),len(fs)
    chi2=np.zeros((nr,npa,nf))

    kerphmod=np.load(gridName+"_kers.npy")   
    for i in range(nr):
        for j in range(npa):
            for k in range(nf):
                chi2[i,j,k]=np.sum(((kerphdat-kerphmod[i,j,k])/sig)**2.)
    minchi2=np.min(chi2)
    print("min chi2 = ",minchi2)
    rmi,pami,fmi = np.where(chi2==minchi2)
    rm,pam,fm = rs[rmi],pas[pami],fs[fmi]
    print("r = ",rm)
    print("pa = ",np.degrees(pam))
    print("f = ",fm)
    return rm,pam,fm,chi2

