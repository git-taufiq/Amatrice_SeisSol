import numpy as np

def smoothspecV(filename):
    seismograms=np.loadtxt(filename)
    dt=seismograms[1,0]
    N=seismograms[:,0].size
    NS=seismograms[0,:].size-1
    T=seismograms[-1,0]
    df=1./T
    f=np.arange(N/2-1)*df
    sf=abs(np.fft.fft(seismograms[:,1:],axis=0))*dt
    sfspecV=sf[:int(N/2+1),:]

    Nfsmooth=100
    flo=0.01
    fro=10
    WB=np.zeros((int(N/2+1),Nfsmooth))
    freqaxis=np.zeros(Nfsmooth)
    sfspecVsmooth=np.zeros((NS,Nfsmooth))
    for j in range(Nfsmooth):
        freqaxis[j]=np.power(10.,(np.log10(fro)-np.log10(flo))/float(Nfsmooth-1)*float(j)+np.log10(flo))
        for i in range(1,int(N/2+1)):
            freq=df*float(i)
            if(freq!=freqaxis[j]):
                WB[i,j]=np.power(np.sin(20.*np.log10(freq/freqaxis[j]))/20./np.log10(freq/freqaxis[j]),4)
            else:
                WB[i,j]=1.
    for j in range(Nfsmooth):
        for i in range(NS):
            sfspecVsmooth[i,j]=sum(abs(sfspecV[0:int(N/2),i])*WB[0:int(N/2),j])/sum(WB[0:int(N/2),j])
    return np.vstack([freqaxis,sfspecVsmooth]).T

def smoothspecA(filename):
    seismograms=np.loadtxt(filename)
    dt=seismograms[1,0]
    N=seismograms[:,0].size
    NS=seismograms[0,:].size-1
    T=seismograms[-1,0]
    df=1./T
    f=np.arange(N/2-1)*df
    sf=abs(np.fft.fft(seismograms[:,1:],axis=0))*dt
    sfspecV=sf[:int(N/2+1),:]

    Nfsmooth=100
    flo=0.01
    fro=10
    WB=np.zeros((int(N/2+1),Nfsmooth))
    freqaxis=np.zeros(Nfsmooth)
    sfspecVsmooth=np.zeros((NS,Nfsmooth))
    for j in range(Nfsmooth):
        freqaxis[j]=np.power(10.,(np.log10(fro)-np.log10(flo))/float(Nfsmooth-1)*float(j)+np.log10(flo))
        for i in range(1,int(N/2+1)):
            freq=df*float(i)
            if(freq!=freqaxis[j]):
                WB[i,j]=np.power(np.sin(20.*np.log10(freq/freqaxis[j]))/20./np.log10(freq/freqaxis[j]),4)
            else:
                WB[i,j]=1.
    for j in range(Nfsmooth):
        for i in range(NS):
            sfspecVsmooth[i,j]=sum(abs(sfspecV[0:int(N/2),i])*WB[0:int(N/2),j])/sum(WB[0:int(N/2),j])
    sfspecAsmooth=(2*np.pi*freqaxis)*sfspecVsmooth
    return np.vstack([freqaxis,sfspecAsmooth]).T
