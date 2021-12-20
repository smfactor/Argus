import numpy as np

def singleAxy(p,SGwin=np.degrees(25.*1.7e-6/2.4)*3600/2.):
    '''
    Returns a function which gives the phase of the complex visibility of a single offset point source at a given point in u,v space. p is relative to the center of the image.

    :param p:
    list of point source parameters.
    p[0]= x offset [arc sec]
    p[1]= y offset [arc sec]
    '''

    xrad=np.radians(p[0]/3600.)
    yrad=np.radians(p[1]/3600.)
    r = np.sqrt(xrad*xrad+yrad*yrad)

    wrad = np.radians(SGwin/3600.)
    b = np.exp(r**4./(-2.*wrad**4))
    def phase(u,v):
        vis=b*np.exp(-2.*np.pi*1.j*(xrad*u+yrad*v))
        return np.angle(vis)
    return phase

def singleArpa(p):
    '''
    Returns a function which gives the phase of the complex visibility of a single offset point source at a given point in u,v space. p is relative to the center of the image.
    
    :param p:
    list of point source parameters.
    p[0]=separation [arc sec]
    p[1]=position angle [radians] E of N
    '''

    xrad=np.radians((p[0]*np.cos(p[1]+np.pi/2.))/3600.)
    yrad=np.radians((p[0]*np.sin(p[1]+np.pi/2.))/3600.)
    def phase(u,v):
        vis=np.exp(-2.*np.pi*1.j*(xrad*u+yrad*v))
        return np.angle(vis)
    return phase

def binaryim(p,shape,ps):
    '''
    Creates a model image with a certain shape and pixel scale of a given number of point sources. One source is at the center, the locations/brightness of the others is given in p.
    Returns the model image.
    NOTE: This assumes one source is at the center. 
    
    :param p:
    list of point source parameters. size(p)/3+1 is the number of sources. 
    p[0]=separation [arc sec]
    p[1]=position angle [radians]
    p[2]=contrast ratio (brightness=1/p[2])
    etc. mod 3
    
    :param shape:
    Tuple of the shape of the image.
    
    :param ps:
    pixel scale in arcsec.

    assumes brightest source is in center of image (false)
    '''
    
    center = shape[0]/2
    modim = np.zeros(shape)
    modim[center,center]=1.

    n = np.size(p)
    for i in np.arange(0,n,3):
        #PA from +y -> -x so y=cos x=-sin
        r=p[i]
        loc = (center + (r*np.cos(p[i+1])/ps), center-(r*np.sin(p[i+1])/ps))
        modim[loc] = 1./p[i+2]
    return modim

def binaryAxy(p):
    '''
    Returns a function which gives the phase of the complex visibility of two point sources at a given point in u,v space. Image is centered on flux centroid NOT the primary. p is still relative to brightest star.
    Only works for a binary (not a tripple).

    :param p:
    list of point source parameters.
    p[0]= x offset [arc sec]
    p[1]= y offset [arc sec]
    p[2]=contrast ratio (brightness of secondary relative to primary =1/p[2])
    '''
    #r0 = np.sqrt(p[0]*p[0]+p[1]*p[1])
    #th0 = np.arctan2(p[1],p[0])
    #p0=[r0,th0-np.pi/2.,p[2]]
    #return binaryArpa(p0)
    r0 = np.sqrt(p[0]*p[0]+p[1]*p[1])
    th0 = np.arctan2(p[1],p[0])
    p0=[r0,th0+np.pi/2.,p[2]]
    d = np.radians(r0/3600.)
    d1 = (1.-(p[2]/(1.+p[2])))*d
    d2 = (p[2]/(1.+p[2]))*d
    def phase(u,v):
        #polar
        #phi = np.arctan2(v,u)
        #r = np.sqrt(u*u+v*v)
        th = np.sqrt(u*u+v*v)*np.cos(np.arctan2(v,u)-th0)
        vis = np.exp(-2.*np.pi*1.j*-d1*th)
        vis+=(1./p[2])*np.exp(-2.*np.pi*1.j*d2*th)
        return np.angle(vis)
    return phase

def binaryArpa(p):
    '''
    Returns a function which gives the phase of the complex visibility of two point sources at a given point in u,v space. Image is centered on flux centroid NOT the primary. p is still relative to brightest star.
    Only works for a binary (not a tripple).
    
    :param p:
    list of point source parameters.
    p[0]=separation [arc sec]
    p[1]=position angle [radians] E of N
    p[2]=contrast ratio (brightness of secondary relative to primary =1/p[2])
    '''
    #th0=p[1]+np.pi/2. #convert PA to regular angle
    #p0=[p[0]*np.cos(th0),p[0]*np.sin(th0),p[2]]
    #return binaryAxy(p0)
    d = np.radians(p[0]/3600.)
    d1 = (1.-(p[2]/(1.+p[2])))*d
    d2 = (p[2]/(1.+p[2]))*d
    def phase(u,v):
        #polar
        phi = np.arctan2(v,u)
        r = np.sqrt(u*u+v*v)
        th = r*np.cos(phi-p[1]-np.pi/2.)
        vis = np.exp(-2.*np.pi*1.j*-d1*th)
        vis+=(1./p[2])*np.exp(-2.*np.pi*1.j*d2*th)
        return np.angle(vis)
    return phase

def binaryAxyPos(p):
    '''
    Returns a function which gives the phase of the complex visibility of two point sources at a given point in u,v space. Image is centered on flux centroid pluss some fitted offset. p is relative to brightest star.

    :param p:
    list of point source parameters.
    p[0]= x offset of image center from flux centroid
    p[1]= y offset of image center from flux centroid
    p[2]= x offset [arc sec]
    p[3]= y offset [arc sec]
    p[4]=contrast ratio (brightness of secondary relative to primary =1/p[2])
    '''
    xrad=np.radians(p[0]/3600.)
    yrad=np.radians(p[1]/3600.)
    r0 = np.sqrt(p[2]*p[2]+p[3]*p[3])
    th0 = np.arctan2(p[3],p[2])
    d = np.radians(r0/3600.)
    d1 = (1.-(p[4]/(1.+p[4])))*d
    d2 = (p[4]/(1.+p[4]))*d
    def phase(u,v):
        #polar
        #phi = np.arctan2(v,u)
        #r = np.sqrt(u*u+v*v)
        th = np.sqrt(u*u+v*v)*np.cos(np.arctan2(v,u)-th0)
        vis = np.exp(-2.*np.pi*1.j*-d1*th)
        vis+=(1./p[4])*np.exp(-2.*np.pi*1.j*d2*th)
        vis*=np.exp(-2*np.pi*1.j*(xrad*u+yrad*v))
        return np.angle(vis)
    return phase

def binaryArpaPos(p,SGwin=np.degrees(25.*1.7e-6/2.4)*3600/2.):
    '''
    Returns a function which gives the phase of the complex visibility of two point sources at a given point in u,v space. Image is centered on flux centroid pluss some fitted offset. p is relative to brightest star.

    :param p:
    list of point source parameters.
    p[0]= x shift of centroid [arc sec]
    p[1]= y shift of centroid [arc sec]
    p[2]= separation from center of image to flux centroid [arcsec]
    p[3]= position angle from center of image to flux centroid [radians]
    p[4]= contrast ratio (brightness of secondary relative to primary =1/p[4])

    :param SGwin:
    width (sigma) of the super-Gaussian window [arcsec].
    '''
    #th0=p[1]+np.pi/2. #convert PA to regular angle
    #th1=p[4]+np.pi/2. #convert PA to regular angle
    #p0=[p[0]*np.cos(th0),p[0]*np.sin(th0),p[2],p[3]*np.cos(th1),p[3]*np.sin(th1)]
    #return binaryAxyPos(p0)
    xrad=np.radians(p[0]/3600.)
    yrad=np.radians(p[1]/3600.)
    d = np.radians(p[2]/3600.)
    d1 = (1.-(p[4]/(1.+p[4])))*d
    d2 = (p[4]/(1.+p[4]))*d

    wrad = np.radians(SGwin/3600.)
    r1 = np.sqrt((xrad+d1*np.sin(p[3]))**2+(yrad+d1*np.cos(p[3]))**2)
    r2 = np.sqrt((xrad-d2*np.sin(p[3]))**2+(yrad-d2*np.cos(p[3]))**2)
    b1 = np.exp(r1**4./(-2.*wrad**4))
    b2 = np.exp(r2**4./(-2.*wrad**4))/p[4]

    def phase(u,v):
        #polar
        #phi = np.arctan2(v,u)
        #r = np.sqrt(u*u+v*v)
        th = np.sqrt(u*u+v*v)*np.cos(np.arctan2(v,u)-p[3]-np.pi/2.)
        vis = b1*np.exp(-2.*np.pi*1.j*-d1*th)
        vis+= b2*np.exp(-2.*np.pi*1.j*d2*th)
        vis*= np.exp(-2*np.pi*1.j*(xrad*u+yrad*v))
        return np.angle(vis)
    return phase


def polyAxy(p,ndim):
    '''
    Returns a function which gives the phase of the complex visibility of a given number of point sources at a given point in u,v space. Image is centered on flux centroid. 

    :param p:
    list of point source parameters. size(p)/3+1 is the number of sources. All parameters are relative to the primary.
    p[0]= x offset [arc sec]
    p[1]= y offset [arc sec]
    p[2]=contrast ratio (brightness=1/p[2])
    etc. mod 3
    '''
    #find centroid
    xc=0.
    yc=0.
    f=1.
    for i in np.arange(0,ndim,3):
        xc+=p[i]/p[i+2]
        yc+=p[i+1]/p[i+2]
        f+=1./p[i+2]
    xc/=f
    yc/=f

    #shift relative to centroid and convert to radians
    p0=np.array([np.radians(-xc/3600.),np.radians(-yc/3600.),1.])
    for i in np.arange(0,ndim,3):
        #axis=0???
        p0=np.append(p0,[np.radians((p[i]-xc)/3600.),np.radians((p[i+1]-yc)/3600.),p[i+2]])

    def phase(u,v):
        r=np.sqrt(u*u+v*v)
        phi=np.arctan2(v,u)
        vis=0.
        for i in np.arange(0,ndim+3,3):
            #dist from O along pa of source 
            #####what is this??????#####
            #th=u*p0[i]+v*p0[i+1]
            th0=np.arctan2(p0[i+1],p0[i])
            th = r*np.cos(phi-th0)
            #source offset
            d=np.sqrt(p0[i]*p0[i]+p0[i+1]*p0[i+1])
            ##### why -1 and not -2???####
            vis+=(1/p0[i+2])*np.exp(-2.*np.pi*1.j*d*th)
        return np.angle(vis)
    return phase

def polyArpa(p,ndim):
    '''
    Returns a function which gives the phase of the complex visibility of a given number of point sources at a given point in u,v space. Image is centered on flux centroid. 

    :param p:
    list of point source parameters. size(p)/3+1 is the number of sources. All parameters are relative to the primary.
    p[0]=separation [arc sec]
    p[1]=position angle [radians] E of N
    p[2]=contrast ratio (brightness=1/p[2])
    etc. mod 3
    '''
    ##convert to xy coords
    ##this doesn't work with pointers (Multinest)
    #p0=np.zeros(ndim)
    #p0[0:ndim:3]=p[0:ndim:3]*np.cos(p[1:ndim:3]+np.pi/2.)
    #p0[1:ndim:3]=p[0:ndim:3]*np.sin(p[1:ndim:3]+np.pi/2.)
    #p0[2:ndim:3]=p[2:ndim:3]
    ##return polyAxy(p0)

    ##find centroid
    #xc=0.
    #yc=0.
    #f=1.
    #for i in np.arange(0,ndim,3):
    #    xc+=p0[i]/p0[i+2]
    #    yc+=p0[i+1]/p0[i+2]
    #    f+=1./p0[i+2]
    #xc/=f
    #yc/=f

    #find centroid in polar coords
    rc=0.
    tc=0.
    f=1.
    for i in np.arange(0,ndim,3):
        rtemp = np.sqrt(rc*rc+(p[i]*p[i])/(p[i+2]*p[i+2])+2.*rc*p[i]*np.cos(p[i+1]-tc)/p[i+2])
        tc = tc + np.arctan2(p[i]*np.sin(p[i+1]-tc)/p[i+2],rc+p[i]*np.cos(p[i+1]-tc)/p[i+2])
        rc = rtemp
        f+=1./p[i+2]
    rc/=f

    ##shift relative to centroid and convert to radians
    #p1=np.array([np.radians(-xc/3600.),np.radians(-yc/3600.),1.])
    #for i in np.arange(0,n,3):
    #    p1=np.append(p1,[np.radians((p0[i]-xc)/3600.),np.radians((p0[i+1]-yc)/3600.),p0[i+2]])

    #shift relative to centroid in polar coords
    ##### create p1 array and fill it instead of append
    p1=np.array([np.radians(rc/3600.),tc+np.pi,1.])
    for i in np.arange(0,ndim,3):
        p1=np.append(p1,[np.radians(np.sqrt(p[i]*p[i]+rc*rc+2.*p[i]*rc*np.cos(tc-p[i+1]-np.pi)/3600.)),p[i+1]+np.arctan2(rc*np.sin(tc-p[i+1]-np.pi),p[i]+rc*np.cos(tc-p[i+1]-np.pi)),p[i+2]])

    def phase(u,v):
        r=np.sqrt(u*u+v*v)
        phi=np.arctan2(v,u)
        vis=0.
        for i in np.arange(0,ndim+3,3):
            #dist from O along pa of source 
            #####what is this??????#####
            #th=u*p1[i]+v*p1[i+1]
            th0=np.arctan2(p1[i+1],p1[i])
            th = r*np.cos(phi-th0)
            #source offset
            d=np.sqrt(p1[i]*p1[i]+p1[i+1]*p1[i+1])
            ##### why -1 and not -1???####
            vis+=(1/p1[i+2])*np.exp(-2.*np.pi*1.j*d*th)
        return np.angle(vis)
    return phase

def polyAxyPos(p,ndim):
    '''
    Returns a function which gives the phase of the complex visibility of a given number of point sources at a given point in u,v space. Image is centered on flux centroid. 

    :param p:
    list of point source parameters. (size(p)-2)/3 + 1 is the number of sources. All parameters are relative to the primary.
    p[0]= x offset [arc sec]
    p[1]= y offset [arc sec]
    p[2]=contrast ratio (brightness=1/p[2])
    etc. mod 3
    p[n-2]= x offset of image center from flux centroid 
    p[n-1]= y offset of image center from flux centroid
    '''

    n=ndim-2
    xrad=np.radians(p[n-2]/3600.)
    yrad=np.radians(p[n-1]/3600.)
    #find centroid
    xc=0.
    yc=0.
    f=1.
    for i in np.arange(0,n,3):
        xc+=p[i]/p[i+2]
        yc+=p[i+1]/p[i+2]
        f+=1./p[i+2]
    xc/=f
    yc/=f

    #shift relative to centroid and convert to radians
    p0=np.array([np.radians(-xc/3600.),np.radians(-yc/3600.),1.])
    for i in np.arange(0,n,3):
        p0=np.append(p0,[np.radians((p[i]-xc)/3600.),np.radians((p[i+1]-yc)/3600.),p[i+2]])

    def phase(u,v):
        r=np.sqrt(u*u+v*v)
        phi=np.arctan2(v,u)
        vis=0.
        for i in np.arange(0,n+3,3):
            #dist from O along pa of source 
            #####what is this??????#####
            #th=u*p0[i]+v*p0[i+1]
            th0=np.arctan2(p0[i+1],p0[i])
            th = r*np.cos(phi-th0)
            #source offset
            d=np.sqrt(p0[i]*p0[i]+p0[i+1]*p0[i+1])
            vis+=(1/p0[i+2])*np.exp(-2.*np.pi*1.j*d*th)
        vis*=np.exp(-2.*np.pi*1.j*(xrad*u+yrad*v))
        return np.angle(vis)
    return phase

def polyArpaPos(p,ndim):
    '''
    Returns a function which gives the phase of the complex visibility of a given number of point sources at a given point in u,v space. Image is centered on flux centroid. 

    :param p:
    list of point source parameters. (size(p)-2)/3 + 1 is the number of sources. All parameters are relative to the primary.
    p[0]=separation [arc sec]
    p[1]=position angle [radians] E of N
    p[2]=contrast ratio (brightness=1/p[2])
    etc. mod 3
    p[n-2]= separation from center of image to flux centroid [arcsec]
    p[n-1]= position angle from center of image to flux centroid [radians]
    '''

    ###This is really slow but I dont know of another way to change coordinates with p as a pointer
    x=p[0]*np.cos(p[1]+np.pi/2.)
    y=p[0]*np.sin(p[1]+np.pi/2.)
    c=p[2]
    p0=np.array([x,y,c])
    for i in np.arange(3,ndim-2,3):
        p0=np.append(p0,[p[i]*np.cos(p[i+1]+np.pi/2.)],axis=0)
        p0=np.append(p0,[p[i]*np.sin(p[i+1]+np.pi/2.)],axis=0)
        p0=np.append(p0,[p[i+2]],axis=0)
        #p0[i]=p[i]*np.cos(p[i+1]+np.pi/2.)
        #p0[i+1]=p[i]*np.sin(p[i+1]+np.pi/2.)
        #p0[i+2]=p[i+2]
    #p0[ndim-2]=p[ndim-2]*np.cos(p[ndim-1]+np.pi/2.)
    #p0[ndim-1]=p[ndim-2]*np.sin(p[ndim-1]+np.pi/2.)
    p0=np.append(p0,[p[ndim-2]*np.cos(p[ndim-1]+np.pi/2.)],axis=0)
    p0=np.append(p0,[p[ndim-2]*np.sin(p[ndim-1]+np.pi/2.)],axis=0)
    #return polyAxyPos(p0)

    n=ndim-2
    #offset
    xrad=np.radians(p0[n-2]/3600.)
    yrad=np.radians(p0[n-1]/3600.)
    #find centroid
    xc=0.
    yc=0.
    f=1.
    for i in np.arange(0,n,3):
        xc+=p0[i]/p0[i+2]
        yc+=p0[i+1]/p0[i+2]
        f+=1./p0[i+2]
    xc/=f
    yc/=f

    #shift relative to centroid and convert to radians
    p1=np.array([np.radians(-xc/3600.),np.radians(-yc/3600.),1.])
    for i in np.arange(0,n,3):
        p1=np.append(p1,[np.radians((p0[i]-xc)/3600.),np.radians((p0[i+1]-yc)/3600.),p0[i+2]])

    def phase(u,v):
        r=np.sqrt(u*u+v*v)
        phi=np.arctan2(v,u)
        vis=0.
        for i in np.arange(0,n+3,3):
            #dist from O along pa of source 
            #####what is this??????#####
            #th=u*p0[i]+v*p0[i+1]
            th0=np.arctan2(p1[i+1],p1[i])
            th = r*np.cos(phi-th0)
            #source offset
            d=np.sqrt(p1[i]*p1[i]+p1[i+1]*p1[i+1])
            vis+=(1/p1[i+2])*np.exp(-2.*np.pi*1.j*d*th)
        vis*=np.exp(-2.*np.pi*1.j*(xrad*u+yrad*v))
        return np.angle(vis)
    return phase




#def modelAsxy(p):
#    '''
#    Returns a function which gives the phase of the complex visibility of a given number of point sources at a given point in u,v space. Now assume image is centered on flux centroid NOT brightest star. p is still relative to brightest star.
#    Only works for a binary (not a tripple).
#    uses sign of p[2] to flip the direction of x, y
#    '''
#    A = 10.**np.abs(p[2])
#    x=p[0]/np.sign(p[2])
#    y=p[1]/np.sign(p[2])
#    r0 = np.sqrt(x*x+y*y)
#    th0 = np.arctan2(y,x)
#    d = np.radians(r0/3600.)
#    d1 = (1.-(A/(1.+A)))*d
#    d2 = (A/(1.+A))*d
#    def phase(u,v):
#        #polar
#        #phi = np.arctan2(v,u)
#        #r = np.sqrt(u*u+v*v)
#        th = np.sqrt(u*u+v*v)*np.cos(np.arctan2(v,u)-th0)
#        vis = np.exp(-2.*np.pi*1.j*-d1*th)
#        vis+=(1./A)*np.exp(-2.*np.pi*1.j*d2*th)
#        ph = np.angle(vis)
#        return ph
#    return phase

#def polyAcen(p):
#    '''
#    Same as modelim(p,shape,ps) but uses analytic FT.
#    Returns a function which gives the phase of the complex visibility of a given number of point sources at a given point in u,v space.
#    assumes brightest source is in center of image (false)
#    '''
#    nsource=np.size(p)/3
#    def phase(u,v):
#        #polar
#        phi = np.arctan2(v,u)
#        r = np.sqrt(u*u+v*v)
#        #distance from 0 at angle of binary + 90degrees (angle of cos wave)
#        #th = r*np.sin(np.pi/2.-p[1]+phi)
#        #d = np.radians(p[0]r3600.)
#        #A = 1./p[2]
#        #re = 1.+A*np.cos(-2.*np.pi*d*th)
#        #im = A*np.sin(-2.*np.pi*d*th)
#        vis=1.
#        for n in range(nsource):
#            i = n*3
#            th = r*np.sin(phi-p[i+1])
#            d = np.radians(p[i]/3600.)
#            A = 1./(p[i+2])
#            vis+=A*np.exp(-2.*np.pi*1.j*d*th)
#            #re += A*np.cos(-2.*np.pi*d*th)
#            #im += A*np.sin(-2.*np.pi*d*th)
#        #ph = np.arctan2(im,re)
#        #ph = np.angle(re+1.j*im)
#        return np.angle(vis)
#    return phase
