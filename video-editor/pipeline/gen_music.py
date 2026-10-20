import numpy as np, soundfile as sf, sys
from scipy.signal import butter, sosfilt
SR=48000; dur=float(sys.argv[2]); bpm=96; beat=60/bpm; n=int(SR*dur); rng=np.random.default_rng(3)
t=np.arange(n)/SR; L=np.zeros(n); R=np.zeros(n)
def lp(x,f): return sosfilt(butter(2,f,'low',fs=SR,output='sos'),x)
def hp(x,f): return sosfilt(butter(2,f,'high',fs=SR,output='sos'),x)
def saw(f,tt): return 2*((f*tt)%1)-1
mid=lambda m:440*2**((m-69)/12)
# chords per 2 bars: Am9, Fmaj7, Cmaj7, Em7 (cold, minor-ish)
chords=[[57,60,64,67,71],[53,57,60,64,69],[48,55,59,64,67],[52,55,59,62,67]]
roots=[45,41,48,40]
bar=4*beat; seg=2*bar
for ci in range(int(dur/seg)+1):
    c=chords[ci%4]; s0=int(ci*seg*SR); s1=min(n,int((ci+1)*seg*SR))
    if s0>=n: break
    tt=np.arange(s1-s0)/SR; ln=len(tt)
    env=np.minimum(1,tt/0.8)*np.minimum(1,(ln/SR-tt)/0.3)
    padL=sum(saw(mid(m)*1.003,tt) for m in c); padR=sum(saw(mid(m)*0.997,tt) for m in c)
    L[s0:s1]+=lp(padL,1400)*env*0.05; R[s0:s1]+=lp(padR,1400)*env*0.05
    # sub bass pulses on 8ths
    f=mid(roots[ci%4]-12)
    for k in range(16):
        b0=s0+int(k*beat/2*SR); b1=min(s1,b0+int(beat/2*SR*0.9))
        if b0>=s1: break
        bt=np.arange(b1-b0)/SR; v=np.sin(2*np.pi*f*bt)*np.exp(-bt/0.25)*0.22*(1 if k%2==0 else 0.6)
        L[b0:b1]+=v; R[b0:b1]+=v
# drums
nb=int(dur/beat)+1
for b in range(nb):
    s=int(b*beat*SR)
    if s>=n: break
    if b%4 in (0,2):
        kt=np.arange(min(int(0.35*SR),n-s))/SR; f=50+90*np.exp(-kt*25); k=np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-kt/0.12)*0.45
        L[s:s+len(kt)]+=k; R[s:s+len(kt)]+=k
    for h in (0,0.5):
        hs=s+int(h*beat*SR); hl=min(int(0.05*SR),n-hs)
        if hl<=0: continue
        ht=np.arange(hl)/SR; hh=hp(rng.standard_normal(hl),7000)*np.exp(-ht/0.015)*(0.10 if h else 0.06)
        L[hs:hs+hl]+=hh*0.8; R[hs:hs+hl]+=hh
    if b%4==3:
        cs=s; cl=min(int(0.2*SR),n-cs); ct=np.arange(cl)/SR
        cl_=sosfilt(butter(2,[1200,5000],'band',fs=SR,output='sos'),rng.standard_normal(cl))*np.exp(-ct/0.05)*0.18
        L[cs:cs+cl]+=cl_; R[cs:cs+cl]+=cl_
# sidechain-ish pump
pump=1-0.35*np.exp(-((t%beat))/0.12); L*=pump; R*=pump
y=np.stack([L,R],1); y/=np.max(np.abs(y))+1e-9; y*=0.7
fi=int(1.0*SR); y[:fi]*=np.linspace(0,1,fi)[:,None]
sf.write(sys.argv[1],y.astype(np.float32),SR); print('music ok',dur)
