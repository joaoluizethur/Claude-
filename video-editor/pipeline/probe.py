import sys, numpy as np, soundfile as sf, librosa, sherpa_onnx
P,E=sys.argv[1],sys.argv[2]; regions=[tuple(map(float,r.split(':'))) for r in sys.argv[3:]]
a,sr=sf.read(f'{P}/a16.wav',dtype='float32'); hop=160
db=20*np.log10(librosa.feature.rms(y=a,frame_length=480,hop_length=hop)[0]+1e-9)
d=f'{E}/sherpa-onnx-whisper-turbo/'
rec=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'turbo-encoder.int8.onnx',decoder=d+'turbo-decoder.int8.onnx',tokens=d+'turbo-tokens.txt',language='pt',task='transcribe',num_threads=4)
def tr(s0,s1):
    st=rec.create_stream(); st.accept_waveform(sr,a[int(s0*sr):int(s1*sr)]); rec.decode_stream(st); return st.result.text.strip()
for r0,r1 in regions:
    seg=db[int(r0*100):int(r1*100)]; thr=np.percentile(seg,30)
    low=seg<thr; dips=[]; i=0
    while i<len(low):
        if low[i]:
            j=i
            while j<len(low) and low[j]: j+=1
            if j-i>=4: dips.append(round(r0+(i+j)/200,2))
            i=j
        else: i+=1
    pts=[r0]+[x for x in dips if x-r0>0.3 and r1-x>0.3]+[r1]
    # merge to >=0.6s pieces
    m=[pts[0]]
    for x in pts[1:]:
        if x-m[-1]>=0.6 or x==r1: m.append(x)
    print(f'== {r0}-{r1}')
    for s0,s1 in zip(m[:-1],m[1:]): print(f'  {s0:7.2f}-{s1:7.2f} | {tr(s0,s1)}',flush=True)
