import numpy as np
from scipy import interpolate

# load traction from stress tensor projection
data = np.genfromtxt('RF_T0m14S.csv', delimiter=',',skip_header=1)
x = data[:,-3]
y = data[:,-2]
z = data[:,-1]
T_s = data[:,3]
T_d = data[:,4]
P_n = data[:,5]
Ts0 = data[:,9]
Td0 = data[:,10]
Pn0 = data[:,11]
xi = np.linspace(-18000,12000,601)
zi = np.linspace(0,-14000*np.sin(45*np.pi/180),281)
from scipy.interpolate import griddata
xi,zi = np.meshgrid(xi,zi)
T_si = checknan(xi,zi,griddata((x, z),-T_s, (xi,zi), method='linear')).T
T_di = checknan(xi,zi,griddata((x, z),-T_d, (xi,zi), method='linear')).T
P_ni = checknan(xi,zi,griddata((x, z), P_n, (xi,zi), method='linear')).T
Ts0i = checknan(xi,zi,griddata((x, z), Ts0, (xi,zi), method='linear')).T
Pn0i = checknan(xi,zi,griddata((x, z),-Pn0, (xi,zi), method='linear')).T
Td0i = checknan(xi,zi,griddata((x, z), Td0, (xi,zi), method='linear')).T

# load from .dat file
forwardmodel = np.loadtxt('forwardmodel.dat')
forwardmodel = forwardmodel[2:] # first two are dummies

# convert to 13 (along strike) x 8 (along dip) sparse grid
T0I = forwardmodel[0:104]
fsI = forwardmodel[104:208]
DcI = forwardmodel[208:312]

T0I_grid = np.zeros((13,8))
fsI_grid = np.zeros((13,8))
DcI_grid = np.zeros((13,8))
for i in range(13):
    T0I_grid[i] = (np.flipud((T0I[i::13]).T)).T
    fsI_grid[i] = (np.flipud((fsI[i::13]).T)).T
    DcI_grid[i] = (np.flipud((DcI[i::13]).T)).T

# bilinearly interpolate T0I, DcI and fsI
x = np.linspace(0, 30000, 13)
z = np.linspace(0, 14000, 8)
kind = 'linear'
f_T0I = interpolate.interp2d(x, z, T0I_grid.T, kind=kind)
f_DcI = interpolate.interp2d(x, z, DcI_grid.T, kind=kind)
f_fsI = interpolate.interp2d(x, z, fsI_grid.T, kind=kind)
grid = 100
xx = np.arange(0, 30000+grid, grid)
zz = np.arange(0, 14000+grid, grid) # using fault top depth at 0 m
T0I_grid_new = (f_T0I(xx, zz)).T # pre-stress (Tau_i)
DcI_grid_new = (f_DcI(xx, zz)).T # critical distance (Dc)
fsI_grid_new = (f_fsI(xx, zz)).T # static - dynamic friction difference

# set parameters
dip = 45
sI = 8.520 * np.sin(dip*np.pi/180) # normal stress depth gradient 8.520 MPa/km
normalstress1d = np.linspace(zz[0]*sI/1000*1e6,zz[-1]*sI/1000*1e6,zz.shape[0])
normalstress1d[normalstress1d<=0.1e6] = 0.1e6

T0_n = T0I_grid_new * 0
for k in range (0,T0_n.shape[0]):
    T0_n[k] = normalstress1d

fdI = 0.55 # dynamic fric. coeff.
cohesion = 0.0
T0_l = T0I_grid_new * 0
T0_m = T0I_grid_new + T0_n * fdI

# dynamic rupture parameters
fs = fsI_grid_new + fdI # static friction coeff. (fsI = static - dynamic fric. coeff)
mud = fs * 0.0 + fdI # dynamic friction coeff. (0.55)
d_c = DcI_grid_new * 1 # critical distance
coh = fs * cohesion # cohesion

T0_n = Pn0i+P_ni*1 
T0_m = Td0i+T_di*1 
T0_l = Ts0i+T_si*1 
T0 = np.sqrt(T0_m**2+T0_l**2)
strength = T0_n*fs
strength_m = strength * 0
factor = strength * 0
for i in range(T0_m.shape[0]):
    for j in range(T0_m.shape[1]):
        strength_m[i,j]=max(0.0,strength[i,j]-0.5e6)
        if T0[i,j]>strength_m[i,j]:
            factor[i,j]=strength_m[i,j]/T0[i,j]
            T0_m[i,j]=T0_m[i,j]*factor[i,j]
            T0_l[i,j]=T0_l[i,j]*factor[i,j]

# keep overstress only in nucleation patch
T0_m[E<0]+=4e6

# create nc file
a = 601
b = 2
c = 281

tn0_3d = np.zeros((a,b,c))
ts0_3d = np.zeros((a,b,c))
td0_3d = np.zeros((a,b,c))
mus0_3d = np.zeros((a,b,c))
mud0_3d = np.zeros((a,b,c))
d_c0_3d = np.zeros((a,b,c))
coh0_3d = np.zeros((a,b,c))

for i in range (0,b):
    tn0_3d[:,i,:] = -T0_n
    ts0_3d[:,i,:] = T0_l
    td0_3d[:,i,:] = T0_m
    mus0_3d[:,i,:] = fs
    mud0_3d[:,i,:] = mud
    d_c0_3d[:,i,:] = d_c
    coh0_3d[:,i,:] = coh
    
tn0_3d = tn0_3d.T
ts0_3d = ts0_3d.T
td0_3d = td0_3d.T
mus0_3d = mus0_3d.T
mud0_3d = mud0_3d.T
d_c0_3d = d_c0_3d.T
coh0_3d = coh0_3d.T

tn0 = tn0_3d.ravel()
ts0 = ts0_3d.ravel()
td0 = td0_3d.ravel()
mus0 = mus0_3d.ravel()
mud0 = mud0_3d.ravel()
d_c0 = d_c0_3d.ravel()
coh0 = coh0_3d.ravel()

# set NetCDF dimensions
xmin = -18000.0
xmax =  12000.0
ymin = -20000.0
ymax =  20000.0
zmin =  0.0
zmax = -9899.49493661167
x = np.linspace(xmin,xmax,a)
y = np.linspace(ymin,ymax,b)
z = np.linspace(zmin,zmax,c)

fout1 = open('Amatrice_fault_DR_TnTsTd.txt','w')
fout2 = open('Amatrice_fault_DR_MusMudDcCoh.txt','w')

fout1.write('netcdf Amatrice_fault_DR {  \n types:  \n compound fault { \n float T_n; \n float T_s; \n float T_d; \n }; \n')
fout1.write('dimensions: \n x = '+ str(x.shape[0]) +'; \n y = '+ str(y.shape[0]) +'; \n z = '+ str(z.shape[0]) +'; \n')
fout1.write('variables: \n float x(x); \n float y(y); \n float z(z); \n fault data(z,y,x); \n')

fout1.write('data:\n x = ')
ii= 0
for i in x:
    if ii == x.shape[0]-1:
        fout1.write(str(i)+';')
    else:
        fout1.write(str(i)+',')
        ii += 1
        
fout1.write('\n y = ')
ii= 0
for i in y:
    if ii == y.shape[0]-1:
        fout1.write(str(i)+';')
    else:
        fout1.write(str(i)+',')
        ii += 1
    
fout1.write('\n z = ')
ii=0
for i in z:
    if ii == z.shape[0]-1:
        fout1.write(str(i)+';')
    else:
        fout1.write(str(i)+',')
        ii += 1
        
fout1.write('\n data = \n')
ii = 0
for i in range(0,x.shape[0]*y.shape[0]*z.shape[0]):
    if ii < x.shape[0]*y.shape[0]*z.shape[0]-1:
        fout1.write('{'+str(tn0[i])+','+str(ts0[i])+','+str(td0[i])+'}, \n')
    else:
        fout1.write('{'+str(tn0[i])+','+str(ts0[i])+','+str(td0[i])+'}; \n')
    ii += 1
            
fout1.write('}\n')
fout1.close()

fout2.write('netcdf Amatrice_fault_DR {  \n types:  \n compound fault { \n float mu_s; \n float mu_d; \n float d_c; \n float cohesion; \n }; \n')
fout2.write('dimensions: \n x = '+ str(x.shape[0]) +'; \n y = '+ str(y.shape[0]) +'; \n z = '+ str(z.shape[0]) +'; \n')
fout2.write('variables: \n float x(x); \n float y(y); \n float z(z); \n fault data(z,y,x); \n')

fout2.write('data:\n x = ')
ii= 0
for i in x:
    if ii == x.shape[0]-1:
        fout2.write(str(i)+';')
    else:
        fout2.write(str(i)+',')
        ii += 1
        
fout2.write('\n y = ')
ii= 0
for i in y:
    if ii == y.shape[0]-1:
        fout2.write(str(i)+';')
    else:
        fout2.write(str(i)+',')
        ii += 1
    
fout2.write('\n z = ')
ii=0
for i in z:
    if ii == z.shape[0]-1:
        fout2.write(str(i)+';')
    else:
        fout2.write(str(i)+',')
        ii += 1
        
fout2.write('\n data = \n')
ii = 0
for i in range(0,x.shape[0]*y.shape[0]*z.shape[0]):
    if ii < x.shape[0]*y.shape[0]*z.shape[0]-1:
        fout2.write('{'+str(mus0[i])+','+str(mud0[i])+','+str(d_c0[i])+','+str(coh0[i])+'}, \n')
    else:
        fout2.write('{'+str(mus0[i])+','+str(mud0[i])+','+str(d_c0[i])+','+str(coh0[i])+'}; \n')
    ii += 1
            
fout2.write('}\n')
fout2.close()
