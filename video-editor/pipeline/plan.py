# usage: plan.py PROJ ENV  (reads PROJ/segments.json: [[start,end],...] in source seconds)
import sys, json, numpy as np, soundfile as sf, librosa, sherpa_onnx
P,E=sys.argv[1],sys.argv[2]
SEGS=json.load(open(f'{P}/segments.json'))
a,sr=sf.read(f'{P}/a16.wav',dtype='float32'); hop=160
db=20*np.log10(librosa.feature.rms(y=a,frame_length=480,hop_length=hop)[0]+1e-9)
PAD, MINSIL, SIL_DB = 0.08, 0.18, -35
def silences(s0,s1):
    q=db[int(s0*100):int(s1*100)]<SIL_DB; out=[]; i=0
    while i<len(q):
        if q[i]:
            j=i
            while j<len(q) and q[j]: j+=1
            if (j-i)/100>=MINSIL: out.append((s0+i/100,s0+j/100))
            i=j
        else: i+=1
    return out
keeps=[]
for s0,s1 in SEGS:
    cur=s0
    for x0,x1 in silences(s0,s1):
        if x0-cur>0.05: keeps.append([cur, x0+PAD])
        cur=x1-PAD
    if s1-cur>0.05: keeps.append([cur,s1])
# drop leading/trailing silence inside keeps
keeps=[[round(k0,3),round(k1,3)] for k0,k1 in keeps if k1-k0>0.15]
d=f'{E}/sherpa-onnx-whisper-turbo/'
rec=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'turbo-encoder.int8.onnx',decoder=d+'turbo-decoder.int8.onnx',tokens=d+'turbo-tokens.txt',language='pt',task='transcribe',num_threads=4)
def tr(s0,s1):
    st=rec.create_stream(); st.accept_waveform(sr,a[int(s0*sr):int(s1*sr)]); rec.decode_stream(st); return st.result.text.strip()
phrases=[]
for k0,k1 in keeps:
    if k1-k0<=3.2: phrases.append([k0,k1,tr(k0,k1)]); continue
    seg=db[int(k0*100):int(k1*100)]; thr=np.percentile(seg,25); low=seg<thr; dips=[]; i=0
    while i<len(low):
        if low[i]:
            j=i
            while j<len(low) and low[j]: j+=1
            if j-i>=4: dips.append((k0+(i+j)/200,(j-i)))
            i=j
        else: i+=1
    cuts=[k0]
    while k1-cuts[-1]>3.2:
        c=[x for x in dips if cuts[-1]+1.2<=x[0]<=cuts[-1]+3.0 and k1-x[0]>=0.8]
        cuts.append(max(c,key=lambda x:x[1])[0] if c else cuts[-1]+2.5)
    cuts.append(k1)
    for s0,s1 in zip(cuts[:-1],cuts[1:]): phrases.append([round(s0,2),round(s1,2),tr(s0,s1)])
json.dump({'keeps':keeps,'phrases':phrases},open(f'{P}/plan.json','w'),ensure_ascii=False,indent=1)
tot=sum(k1-k0 for k0,k1 in keeps); print('keeps',len(keeps),'total',round(tot,2))
for p in phrases: print(f'{p[0]:7.2f}-{p[1]:7.2f} | {p[2]}')
