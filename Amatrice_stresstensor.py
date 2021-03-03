import numpy as np
from scipy import interpolate

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

# # add percentage of T0_m
# T0_m = T0_m + 0.4 * T0_m

# # add percentage of T0_m only inside slip area
# data = np.genfromtxt('../Data/Case_A_ASl_40s.csv', delimiter=',',skip_header=1)
# x = data[:,-3]
# y = data[:,-2]
# z = data[:,-1]
# ASl = data[:,15]
# xi = np.linspace(-18000,12000,601)
# zi = np.linspace(0,-14000*np.sin(45*np.pi/180),281)
# from scipy.interpolate import griddata
# ASli = griddata((x, z), ASl, (xi[None,:], zi[:,None]), method='linear').T
# SSinv = ASli*1
# Smin = 0.6;Smax = 0.01
# SSinv[SSinv>Smin]=Smin
# SSinv[SSinv<Smax]=Smax
# SSinv = ((SSinv-Smax)/(Smin-Smax))
# T0_m = T0_m + 0.4*T0_m*SSinv

# vertical fault stresses (case: align with x-axis)
stress_vertical = np.zeros((9,T0I_grid_new.shape[0],T0I_grid_new.shape[1]))
stress_vertical[0] =-(T0_n) # S11
stress_vertical[1] =-(T0_n) # S22
stress_vertical[2] =-(T0_n) # S33
stress_vertical[3] =-(T0_l) # S12
stress_vertical[4] =-(T0_m) # S23
stress_vertical[5] =-(T0_l) # S13
stress_vertical[6] =-(T0_l) # S21
stress_vertical[7] =-(T0_l) # S31
stress_vertical[8] =-(T0_m) # S32

# stress tensor rotation 
thetax = 45 # input rotation angles around x-axis
thetay =  0 # input rotation angles around y-axis
thetaz =  0 # input rotation angles around z-axis

# rotary tensor for rotation around x-axis
Tx = np.matrix([[1,                       0,                        0],
                [0,np.cos(thetax*np.pi/180),-np.sin(thetax*np.pi/180)],
                [0,np.sin(thetax*np.pi/180), np.cos(thetax*np.pi/180)]])

# rotary tensor for rotation around y-axis
Ty = np.matrix([[np.cos(thetay*np.pi/180),0,-np.sin(thetay*np.pi/180)],
                [                       0,1,                        0],
                [np.sin(thetay*np.pi/180),0, np.cos(thetay*np.pi/180)]])

# rotary tensor for rotation around z-axis
Tz = np.matrix([[np.cos(thetaz*np.pi/180),-np.sin(thetaz*np.pi/180),0],
                [np.sin(thetaz*np.pi/180), np.cos(thetaz*np.pi/180),0],
                [                       0,                        0,1]])
#   [ sxx sxy sxz ]
# S=[ syx syy syz ]
#   [ szx szy szz ]
sxx_rot = []
syy_rot = []
szz_rot = []
sxy_rot = []
syz_rot = []
sxz_rot = []
for j in range (stress_vertical.shape[2]):
    for i in range (stress_vertical.shape[1]):
        rotX = (Tx @ np.matrix([[stress_vertical[0,i,j],stress_vertical[3,i,j],stress_vertical[5,i,j]],
                                [stress_vertical[6,i,j],stress_vertical[1,i,j],stress_vertical[4,i,j]],
                                [stress_vertical[7,i,j],stress_vertical[8,i,j],stress_vertical[2,i,j]]]) @ np.transpose(Tx))
        rotY = (Ty @ rotX @ np.transpose(Ty))
        rotZ = (Tz @ rotY @ np.transpose(Tz))
        sxx_rot.append(rotZ[0,0])
        syy_rot.append(rotZ[1,1])
        szz_rot.append(rotZ[2,2])
        sxy_rot.append(rotZ[0,1])
        syz_rot.append(rotZ[1,2])
        sxz_rot.append(rotZ[0,2])
sxx_rot = np.asarray(sxx_rot)
syy_rot = np.asarray(syy_rot)
szz_rot = np.asarray(szz_rot)
sxy_rot = np.asarray(sxy_rot)
syz_rot = np.asarray(syz_rot)
sxz_rot = np.asarray(sxz_rot)

# using (along strike) grid x (along dip) grid
stress_rot = stress_vertical * 0
stress_rot[0] = np.reshape(sxx_rot,(-1,stress_vertical.shape[1])).T # Sxx
stress_rot[1] = np.reshape(syy_rot,(-1,stress_vertical.shape[1])).T # Syy
stress_rot[2] = np.reshape(szz_rot,(-1,stress_vertical.shape[1])).T # Szz
stress_rot[3] = np.reshape(sxy_rot,(-1,stress_vertical.shape[1])).T # Sxy
stress_rot[4] = np.reshape(syz_rot,(-1,stress_vertical.shape[1])).T # Syz
stress_rot[5] = np.reshape(sxz_rot,(-1,stress_vertical.shape[1])).T # Sxz

a = 601
b = 2
c = 281
sxx0_3d = np.zeros((a,b,c))
syy0_3d = np.zeros((a,b,c))
szz0_3d = np.zeros((a,b,c))
sxy0_3d = np.zeros((a,b,c))
syz0_3d = np.zeros((a,b,c))
sxz0_3d = np.zeros((a,b,c))
mus0_3d = np.zeros((a,b,c))
mud0_3d = np.zeros((a,b,c))
d_c0_3d = np.zeros((a,b,c))
coh0_3d = np.zeros((a,b,c))

for i in range (0,b):
    sxx0_3d[:,i,:] = stress_rot[0]
    syy0_3d[:,i,:] = stress_rot[1]
    szz0_3d[:,i,:] = stress_rot[2]
    sxy0_3d[:,i,:] = stress_rot[3]
    syz0_3d[:,i,:] = stress_rot[4]
    sxz0_3d[:,i,:] = stress_rot[5]
    mus0_3d[:,i,:] = fs
    mud0_3d[:,i,:] = mud
    d_c0_3d[:,i,:] = d_c
    coh0_3d[:,i,:] = coh
    
sxx0_3d = sxx0_3d.T
syy0_3d = syy0_3d.T
szz0_3d = szz0_3d.T
sxy0_3d = sxy0_3d.T
syz0_3d = syz0_3d.T
sxz0_3d = sxz0_3d.T
mus0_3d = mus0_3d.T
mud0_3d = mud0_3d.T
d_c0_3d = d_c0_3d.T
coh0_3d = coh0_3d.T

sxx0 = sxx0_3d.ravel()
syy0 = syy0_3d.ravel()
szz0 = szz0_3d.ravel()
sxy0 = sxy0_3d.ravel()
syz0 = syz0_3d.ravel()
sxz0 = sxz0_3d.ravel()
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

fout = open('Amatrice_Case_A2_sxx/Amatrice_fault_DR_sxx.txt','w')
fout.write('netcdf Amatrice_fault_DR {  \n types:  \n compound fault { \n float s_xx; \n float s_yy; \n float s_zz; \n float s_xy; \n float s_yz;  \n float s_xz; \n float mu_s; \n float mu_d; \n float d_c; \n float cohesion; \n }; \n')
fout.write('dimensions: \n x = '+ str(x.shape[0]) +'; \n y = '+ str(y.shape[0]) +'; \n z = '+ str(z.shape[0]) +'; \n')
fout.write('variables: \n float x(x); \n float y(y); \n float z(z); \n fault data(z,y,x); \n')

fout.write('data:\n x = ')
ii= 0
for i in x:
    if ii == x.shape[0]-1:
        fout.write(str(i)+';')
    else:
        fout.write(str(i)+',')
        ii += 1
        
fout.write('\n y = ')
ii= 0
for i in y:
    if ii == y.shape[0]-1:
        fout.write(str(i)+';')
    else:
        fout.write(str(i)+',')
        ii += 1
    
fout.write('\n z = ')
ii=0
for i in z:
    if ii == z.shape[0]-1:
        fout.write(str(i)+';')
    else:
        fout.write(str(i)+',')
        ii += 1
        
fout.write('\n data = \n')
ii = 0
for i in range(0,x.shape[0]*y.shape[0]*z.shape[0]):
    if ii < x.shape[0]*y.shape[0]*z.shape[0]-1:
        fout.write('{'+str(sxx0[i])+','+str(syy0[i])+','+str(szz0[i])+','+str(sxy0[i])+','+str(syz0[i])+','+str(sxz0[i])+','+str(mus0[i])+','+str(mud0[i])+','+str(d_c0[i])+','+str(coh0[i])+'}, \n')
    else:
        fout.write('{'+str(sxx0[i])+','+str(syy0[i])+','+str(szz0[i])+','+str(sxy0[i])+','+str(syz0[i])+','+str(sxz0[i])+','+str(mus0[i])+','+str(mud0[i])+','+str(d_c0[i])+','+str(coh0[i])+'}; \n')
    ii += 1
            
fout.write('}\n')
fout.close()
