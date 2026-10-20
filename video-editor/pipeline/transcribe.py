import sys, json, numpy as np, soundfile as sf, sherpa_onnx
P=sys.argv[1]; E=sys.argv[2]; msil=float(sys.argv[3])
a,sr=sf.read(f'{P}/a16.wav',dtype='float32')
cfg=sherpa_onnx.VadModelConfig(); cfg.silero_vad.model=f'{E}/silero_vad.onnx'; cfg.silero_vad.min_silence_duration=msil; cfg.silero_vad.min_speech_duration=0.1; cfg.silero_vad.threshold=0.4; cfg.silero_vad.max_speech_duration=8; cfg.sample_rate=sr
vad=sherpa_onnx.VoiceActivityDetector(cfg,buffer_size_in_seconds=60); segs=[]; w=cfg.silero_vad.window_size
def drain():
    while not vad.empty(): segs.append((vad.front.start,len(vad.front.samples))); vad.pop()
for k in range(0,len(a),w): vad.accept_waveform(a[k:k+w]); drain()
vad.flush(); drain()
d=f'{E}/sherpa-onnx-whisper-turbo/'
rec=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'turbo-encoder.int8.onnx',decoder=d+'turbo-decoder.int8.onnx',tokens=d+'turbo-tokens.txt',language='pt',task='transcribe',num_threads=4)
out=[]
for st,n in segs:
    s0=max(0,st-int(0.08*sr)); s1=min(len(a),st+n+int(0.08*sr))
    s=rec.create_stream(); s.accept_waveform(sr,a[s0:s1]); rec.decode_stream(s)
    out.append({'start':round(st/sr,3),'end':round((st+n)/sr,3),'text':s.result.text.strip()})
    print(f"{st/sr:6.2f}-{(st+n)/sr:6.2f} | {out[-1]['text']}",flush=True)
json.dump(out,open(f'{P}/segs_{msil}.json','w'),ensure_ascii=False,indent=1)
