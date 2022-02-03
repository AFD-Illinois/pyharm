#
#
import numpy as np
import matplotlib.pyplot as plt
#
# let's specify a B field that is analytically divergence-free but 
# on a grid has estiamte div.B that is of order the truncation error
#
def blooper(x, y):
    
    rsq = x**2 + y**2
    s = 0.2
    B = rsq*np.exp(-rsq/(2*s**2))
    phi = np.arctan2(y,x) + 0.5*np.pi
    Bx = B*np.cos(phi)
    By = B*np.sin(phi)

    return(Bx, By)
#done

def xy2ij(X,Y,Nx,Ny):
    dx = 1./Nx
    dy = 1./Ny

    hx = Nx/2. - 0.5
    hy = Ny/2. - 0.5

    i = (X/dx) + hx
    j = (Y/dy) + hy
    return(i,j)
#done

def ij2xy(i,j,Nx,Ny):

    dx = 1./Nx
    dy = 1./Ny
    hx = Nx/2. - 0.5
    hy = Ny/2. - 0.5
    x = (i - hx)*dx
    y = (j - hy)*dy

    return(x,y)
#done

def divcalc(Vx, Vy, Nx, Ny):
    dx = 1./Nx
    dy = 1./Ny

    Vx11 = np.roll(Vx, (1,1), axis=(0,1))
    Vx10 = np.roll(Vx, (1,0), axis=(0,1))
    Vx01 = np.roll(Vx, (0,1), axis=(0,1))
    Vx00 = np.roll(Vx, (0,0), axis=(0,1))

    Vy11 = np.roll(Vy, (1,1), axis=(0,1))
    Vy01 = np.roll(Vy, (0,1), axis=(0,1))
    Vy10 = np.roll(Vy, (1,0), axis=(0,1))
    Vy00 = np.roll(Vy, (0,0), axis=(0,1))

    div = 0.5*((Vx11+Vx10-Vx01-Vx00)/dx + (Vy11+Vy01-Vy10-Vy00)/dy)
    return(div)
#done

def LinearInterpolate(f, Nx, Ny, ninterp):
    fH = np.zeros((Nx*ninterp, Ny*ninterp))
    NxH = Nx*ninterp
    NyH = Ny*ninterp
    xH = np.zeros((NxH,NyH))
    yH = np.zeros((NxH,NyH))
    for i in range(NxH):
        for j in range(NyH):

            # what is position on new grid?
            X,Y = ij2xy(i,j,NxH,NyH)
            xH[i,j] = X
            yH[i,j] = Y

            # interpolate from old grid
            io,jo = xy2ij(X,Y,Nx,Ny)
            # 
            ioI = (int(io) + Nx)%Nx 
            joI = (int(jo) + Ny)%Ny
            dio = io-ioI
            djo = jo-joI
            ioIp = (ioI + Nx + 1)%Nx 
            joIp = (joI + Ny + 1)%Ny 

            # bilinear interpolation 
            fH[i,j] = \
                (1. - dio)*(1. - djo)*f[ioI,joI] + \
                dio*(1. - djo)*f[ioIp,joI] + \
                (1. - dio)*djo*f[ioI,joIp] + \
                dio*djo*f[ioIp,joIp]

    return(fH,xH,yH)
#done

def Grad(p, Nx, Ny):
    dx = 1./Nx
    dy = 1./Ny

    # apply grad 
    p11 = np.roll(p, (0,0),   axis=(0,1))  # p is CORNER CENTERED
    p10 = np.roll(p, (0,-1),  axis=(0,1))  # doesn't work w/o axis!
    p01 = np.roll(p, (-1,0),  axis=(0,1))
    p00 = np.roll(p, (-1,-1), axis=(0,1))

    gradpx = 0.5*(p11+p10-p01-p00)/dx # gradp is zone centered
    gradpy = 0.5*(p11+p01-p10-p00)/dy

    return(gradpx, gradpy)
# done

def Laplacian(p, Nx, Ny):

    # apply grad 
    gradpx, gradpy = Grad(p, Nx, Ny)

    # now take divergence
    div = divcalc(gradpx, gradpy, Nx, Ny)
    return(div)
#done

def divclean(Vx, Vy, Nx, Ny):
    dx = 1./Nx
    dy = 1./Ny

    m = divcalc(Vx, Vy, Nx, Ny)
    p = np.copy(m) # initial guess

    errTOL = 1.e-8
    err = 100.*errTOL
    niter = 0

    K = 1.
    dt = 0.25*(1. - np.pi/Nx)*dx*dx/K

    errM = np.zeros((Nx,Ny))  # error array
    while (err > errTOL and niter < 4*Nx*Ny):
        #
        lap = Laplacian(p, Nx, Ny)
        errM = K*lap - m
        dp = errM*dt
        p = p + dp
        #
        err = np.abs( np.sum(np.abs(errM))/np.sum(np.abs(m)) )
        niter=niter+1
        print(niter, err)
    # end while

    # clean up
    gradpx, gradpy = Grad(p, Nx, Ny)
    VxC = Vx - gradpx
    VyC = Vy - gradpy

    return(VxC, VyC)
# done

# create a grid and calculate B on it
nx = 64 
ny = 64
bx = np.zeros((nx, ny))
by = np.zeros((nx, ny))
x = np.zeros((nx, ny))
y = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        X,Y = ij2xy(i,j,nx,ny)
        Bx,By = blooper(X,Y)
        x[i,j] = X
        y[i,j] = Y
        bx[i,j] = Bx
        by[i,j] = By

# done

# show it off
skip = 8
plt.quiver(x[::skip,::skip].T, y[::skip,::skip].T,bx[::skip,::skip].T, by[::skip,::skip].T)
plt.show()
#
plt.imshow(bx, origin='lower')
plt.show()

# now evaluate divb
#print('here we go...')
divb = divcalc(bx, by, nx, ny)
bmean = np.mean(np.sqrt(bx**2 + by**2))
dx = 1./nx
dy = 1./ny
divbnorm = divb*(dx + dy)/bmean
cut = 2
plt.imshow(divbnorm[cut:nx-cut,cut:ny-cut], cmap='afmhot', origin='lower')
plt.show()

# OK, now apply divb cleaner
bxC, byC = divclean(bx, by, nx, ny)

# now look at output
divb = divcalc(bxC, byC, nx, ny)
plt.imshow(divb.T, origin='lower')
plt.show()

# now interpolate
ninterp = 2  
bxH,xH,yH = LinearInterpolate(bxC, nx, ny, ninterp)
byH,xH,yH = LinearInterpolate(byC, nx, ny, ninterp)
nxH = ninterp*nx
nyH = ninterp*ny
skip = 8
plt.quiver(xH[::skip,::skip].T, yH[::skip,::skip].T,bxH[::skip,::skip].T, byH[::skip,::skip].T)
plt.show()

divbH = divcalc(bxH, byH, nxH, nyH)
bmean = np.mean(np.sqrt(bxH**2 + byH**2))
dx = 1./nxH
dy = 1./nyH
divbnorm = divbH*(dx + dy)/bmean
cut = 2
plt.imshow(divbnorm[cut:nx-cut,cut:ny-cut], cmap='afmhot', origin='lower')
plt.show()

# now clean it up
bxHC, byHC = divclean(bxH, byH, nxH, nyH)

# plot
divbHC = divcalc(bxHC, byHC, nxH, nyH)
plt.imshow(divbHC.T, origin='lower')
plt.show()

#
plt.quiver(xH[::skip,::skip].T, yH[::skip,::skip].T,bxHC[::skip,::skip].T, byHC[::skip,::skip].T)
plt.show()

plt.imshow(bxHC.T, origin='lower')
plt.show()


