import numpy as np, soundfile as sf, sys
from scipy.signal import butter, sosfilt
SR=48000; out=sys.argv[1]; rng=np.random.default_rng(7)
def norm(x,db=-3): x=x/np.max(np.abs(x)+1e-9); return (x*10**(db/20)).astype(np.float32)
def st(x): return np.stack([x,x],1)
def whoosh(d=0.25):
    n=int(SR*d); noise=rng.standard_normal(n); y=np.zeros(n); blk=512
    fs=np.geomspace(200,6000,n//blk+1)
    for i in range(0,n,blk):
        f=fs[i//blk]; sos=butter(2,[f*0.7,min(f*1.4,23000)],'band',fs=SR,output='sos'); y[i:i+blk]=sosfilt(sos,noise[i:i+blk])
    t=np.linspace(0,1,n); env=np.sin(np.pi*t**0.6)**2
    return y*env
def pop(d=0.06):
    t=np.arange(int(SR*d))/SR; return np.sin(2*np.pi*900*t)*np.exp(-t/0.012)
def impact(d=0.7):
    t=np.arange(int(SR*d))/SR; f=45*np.exp(-t*1.2)+20; ph=2*np.pi*np.cumsum(f)/SR
    body=np.sin(ph)*np.exp(-t/0.25); n=sosfilt(butter(2,1500,'low',fs=SR,output='sos'),rng.standard_normal(len(t)))*np.exp(-t/0.05)*0.6
    return np.tanh(2*(body+n))
def riser(d=1.5):
    t=np.arange(int(SR*d))/SR; f=120*(3000/120)**(t/d); ph=2*np.pi*np.cumsum(f)/SR
    n=sosfilt(butter(2,[800,8000],'band',fs=SR,output='sos'),rng.standard_normal(len(t)))*0.3
    return (np.sin(ph)*0.6+n)*(t/d)**2
w=whoosh()
for name,x in [('whoosh',w),('pop',pop()),('impact',impact()),('riser',riser()),('swoosh_rev',w[::-1].copy())]:
    sf.write(f'{out}/{name}.wav',st(norm(x)),SR)
print('ok')
