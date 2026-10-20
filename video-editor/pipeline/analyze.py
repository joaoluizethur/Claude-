import sys, json, numpy as np, soundfile as sf, sherpa_onnx, librosa
P=sys.argv[1]; E=sys.argv[2]
a,sr=sf.read(f'{P}/a16.wav',dtype='float32')
# silence map: frames below -35 dBFS for >0.4s
hop=160; rms=librosa.feature.rms(y=a,frame_length=640,hop_length=hop)[0]; db=20*np.log10(rms+1e-9)
quiet=db<-35; sil=[]; i=0
while i<len(quiet):
    if quiet[i]:
        j=i
        while j<len(quiet) and quiet[j]: j+=1
        if (j-i)*hop/sr>=0.4: sil.append((round(i*hop/sr,2),round(j*hop/sr,2)))
        i=j
    else: i+=1
print('peak dBFS', round(db.max(),1), 'median', round(np.median(db),1))
# VAD speech segments
cfg=sherpa_onnx.VadModelConfig(); cfg.silero_vad.model=f'{E}/silero_vad.onnx'; cfg.silero_vad.min_silence_duration=0.18; cfg.silero_vad.min_speech_duration=0.15; cfg.silero_vad.threshold=0.45; cfg.silero_vad.max_speech_duration=12; cfg.sample_rate=sr
vad=sherpa_onnx.VoiceActivityDetector(cfg,buffer_size_in_seconds=60)
segs=[]; w=cfg.silero_vad.window_size
for k in range(0,len(a),w):
    vad.accept_waveform(a[k:k+w])
    while not vad.empty(): segs.append((vad.front.start, len(vad.front.samples))); vad.pop()
vad.flush()
while not vad.empty(): segs.append((vad.front.start, len(vad.front.samples))); vad.pop()
d=f'{E}/sherpa-onnx-whisper-small/'
rec=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'small-encoder.onnx',decoder=d+'small-decoder.onnx',tokens=d+'small-tokens.txt',language='pt',task='transcribe',num_threads=4)
out=[]
for st,n in segs:
    s0=max(0,st-int(0.05*sr)); s1=min(len(a),st+n+int(0.05*sr))
    s=rec.create_stream(); s.accept_waveform(sr,a[s0:s1]); rec.decode_stream(s)
    out.append({'start':round(st/sr,3),'end':round((st+n)/sr,3),'text':s.result.text.strip()})
# energy peaks (onset strength on speech, top moments)
ae,sr2=sf.read(f'{P}/a16.wav',dtype='float32'); rr=librosa.feature.rms(y=ae,frame_length=4096,hop_length=1600)[0]; t=np.arange(len(rr))*0.1
idx=np.argsort(rr)[::-1]; peaks=[]
for i in idx:
    if all(abs(t[i]-p)>2.5 for p in peaks): peaks.append(round(float(t[i]),1))
    if len(peaks)>=6: break
json.dump({'segments':out,'silences':sil,'peaks':sorted(peaks)},open(f'{P}/analysis.json','w'),ensure_ascii=False,indent=1)
for o in out: print(f"{o['start']:6.2f}-{o['end']:6.2f} | {o['text']}")
print('silences',sil); print('peaks',sorted(peaks))
