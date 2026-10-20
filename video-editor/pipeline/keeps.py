import sys, numpy as np, soundfile as sf, librosa, sherpa_onnx
P,E=sys.argv[1],sys.argv[2]
a,sr=sf.read(f'{P}/a16.wav',dtype='float32'); hop=160
db=20*np.log10(librosa.feature.rms(y=a,frame_length=480,hop_length=hop)[0]+1e-9); t=np.arange(len(db))*hop/sr
# print energy 1.0-3.0s in 50ms steps
for x in np.arange(1.0,3.0,0.05): print(f'{x:.2f}:{db[int(x*sr/hop)]:.0f}',end=' ')
print()
# silences >=0.18s below -35
q=db<-35; sil=[]; i=0
while i<len(q):
    if q[i]:
        j=i
        while j<len(q) and q[j]: j+=1
        if (j-i)*hop/sr>=0.18: sil.append((round(t[i],2),round(t[min(j,len(t)-1)],2)))
        i=j
    else: i+=1
print('sil>=0.18',sil)
d=f'{E}/sherpa-onnx-whisper-turbo/'
rec=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'turbo-encoder.int8.onnx',decoder=d+'turbo-decoder.int8.onnx',tokens=d+'turbo-tokens.txt',language='pt',task='transcribe',num_threads=4)
for s0,s1 in [(1.0,2.45),(2.45,4.3),(25.5,27.2),(9.5,11.6)]:
    st=rec.create_stream(); st.accept_waveform(sr,a[int(s0*sr):int(s1*sr)]); rec.decode_stream(st); print(s0,s1,'|',st.result.text)
