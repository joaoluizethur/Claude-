import sys, json, numpy as np, soundfile as sf, librosa, sherpa_onnx
P,E=sys.argv[1],sys.argv[2]
a,sr=sf.read(f'{P}/a16.wav',dtype='float32')
hop=160; db=20*np.log10(librosa.feature.rms(y=a,frame_length=480,hop_length=hop)[0]+1e-9)
# smoothed; find dips: runs below thr >= 50ms
thr=np.percentile(db,25)
low=db<thr; dips=[]; i=0
while i<len(low):
    if low[i]:
        j=i
        while j<len(low) and low[j]: j+=1
        if (j-i)*hop/sr>=0.05: dips.append(((i+j)/2*hop/sr, (j-i)*hop/sr))
        i=j
    else: i+=1
print('thr',round(thr,1),'dips',len(dips))
# choose cut points: greedy, slices between 1.2 and 3.2s, prefer longest dips
cuts=[1.2]; end=30.6
cand=sorted(d[0] for d in dips)
while cuts[-1]<end-1.0:
    c=[d for d in dips if cuts[-1]+1.2<=d[0]<=cuts[-1]+3.2]
    if not c: nxt=min(cuts[-1]+3.0,end)
    else: nxt=max(c,key=lambda d:d[1])[0]
    cuts.append(round(nxt,2))
if cuts[-1]<end: cuts.append(end)
d=f'{E}/sherpa-onnx-whisper-turbo/'
rec=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'turbo-encoder.int8.onnx',decoder=d+'turbo-decoder.int8.onnx',tokens=d+'turbo-tokens.txt',language='pt',task='transcribe',num_threads=4)
out=[]
for s0,s1 in zip(cuts[:-1],cuts[1:]):
    st=rec.create_stream(); st.accept_waveform(sr,a[int(s0*sr):int(s1*sr)]); rec.decode_stream(st)
    out.append([s0,s1,st.result.text.strip()]); print(f"{s0:6.2f}-{s1:6.2f} | {out[-1][2]}",flush=True)
json.dump(out,open(f'{P}/slices.json','w'),ensure_ascii=False)
