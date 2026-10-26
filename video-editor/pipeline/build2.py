# Generic talking-head build. usage: build2.py SCRATCH PROJDIR  (reads PROJDIR/config.json + plan.json)
import json, subprocess, sys, numpy as np, soundfile as sf, pyloudnorm as pyln, librosa
from PIL import ImageFont

S, P = sys.argv[1], sys.argv[2]
E = f'{S}/env'
C = json.load(open(f'{P}/config.json'))
FPS, W, H = 30, 1080, 1920
SRC_W, SRC_H = C['src_size']
ZOOM = 1.12
ZOOMS, CAPTION_ANIM = True, False
MIN_HOLD = C.get('min_zoom_hold', 2.5)  # seconds before framing may alternate again

def fr(x):
    return round(x * FPS) / FPS

keeps = [list(k) for k in json.load(open(f'{P}/plan.json'))['keeps']]
keeps[-1][1] = C.get('last_end', keeps[-1][1])
SHOTS, z, since = [], C.get('first_zoom', 'T'), 0.0
for i, (a, b) in enumerate(keeps):
    if i and since >= MIN_HOLD:
        z, since = ('W' if z == 'T' else 'T'), 0.0
    SHOTS.append((fr(a), fr(b), z if ZOOMS else 'W'))
    since += b - a
SHOTS = [s for s in SHOTS if s[1] > s[0]]
cw, ch = round(SRC_W / ZOOM / 2) * 2, round(SRC_H / ZOOM / 2) * 2
cx, cy = round((SRC_W - cw) / 2), C['tight_y0']

out_starts, acc = [], 0.0
for a, b, _ in SHOTS:
    out_starts.append(acc)
    acc += b - a
TOTAL = acc
TAIL = C['tail']
TOTAL_V = TOTAL + round(TAIL * FPS) / FPS

def to_out(s):
    for (a, b, _), o in zip(SHOTS, out_starts):
        if s < a:
            return o
        if s <= b:
            return o + s - a
    return TOTAL

a16, _ = sf.read(f'{P}/a16.wav', dtype='float32')
db = 20 * np.log10(librosa.feature.rms(y=a16, frame_length=480, hop_length=160)[0] + 1e-9)
def dip_near(t, win=0.2):
    i0, i1 = int((t - win) * 100), int((t + win) * 100)
    return (i0 + int(np.argmin(db[i0:i1]))) / 100

CUES = [(dip_near(t) if snap else t, text, hl) for t, text, hl, snap in
        [(c[0], c[1], c[2], c[3] if len(c) > 3 else False) for c in C['cues']]]

FS = 96
font = ImageFont.truetype(f'{E}/subfonts/MontserratET.ttf', 100)
MAXW = 840

def width(line, fs):
    return font.getlength(line) * fs * 0.64 / 100 + len(line) + 16

def wrap(text):
    words = text.split()
    if width(text, FS) <= MAXW:
        return [words], FS
    best = None
    for k in range(1, len(words)):
        w = max(width(' '.join(words[:k]), FS), width(' '.join(words[k:]), FS))
        if best is None or w < best[0]:
            best = (w, k)
    fs = FS if best[0] <= MAXW else int(FS * MAXW / best[0])
    return [words[:best[1]], words[best[1]:]], fs

def ts(t):
    cs = int(round(t * 100))
    return f'{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}'

def srt_ts(t):
    ms = int(round(t * 1000))
    return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}'

ANIM = '\\fad(60,0)\\fscx90\\fscy90\\t(0,110,\\fscx100\\fscy100)'
HL, BASE = '&H00F0C686&', '&H00EBEBEB&'
ass = [
    '[Script Info]', 'ScriptType: v4.00+', f'PlayResX: {W}', f'PlayResY: {H}', 'WrapStyle: 2', 'ScaledBorderAndShadow: yes', '',
    '[V4+ Styles]',
    'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
    f'Style: Cap,Montserrat ET,{FS},{BASE},{BASE},&H00242424&,&H90000000&,0,0,0,0,100,100,1,0,1,8,4,5,108,108,0,1',
    f'Style: Cta,Montserrat ET,78,{BASE},{BASE},&H00242424&,&H26242424&,0,0,0,0,100,100,1,0,3,26,0,5,108,108,0,1',
    '', '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
]
srt = []
speech_end = to_out(C['speech_end']) + 0.15
cta_t0 = to_out(C['cta_at']) if 'cta_at' in C else speech_end
for i, (s, text, hl) in enumerate(CUES):
    t0 = to_out(s)
    t1 = to_out(CUES[i + 1][0]) if i + 1 < len(CUES) else speech_end
    wrapped, fs = wrap(text)
    body = '\\N'.join(' '.join(f'{{\\c{HL}}}{w}{{\\c{BASE}}}' if w in hl else w for w in line) for line in wrapped)
    ass.append(f'Dialogue: 0,{ts(t0)},{ts(t1)},Cap,,0,0,0,,{{\\an5\\pos(540,1498)\\fs{fs}{ANIM if CAPTION_ANIM else ""}}}{body}')
    srt.append(f'{i + 1}\n{srt_ts(t0)} --> {srt_ts(t1)}\n{text.capitalize()}\n')
CTA = ('ME SIGA', 'PARA MAIS')
ass.append(f'Dialogue: 1,{ts(cta_t0)},{ts(TOTAL_V)},Cta,,0,0,0,,{{\\an5\\move(540,1600,540,1498,0,500)\\fad(500,0)}}{{\\c{HL}}}{CTA[0]}{{\\c{BASE}}} {CTA[1]}')
srt.append(f'{len(CUES) + 1}\n{srt_ts(cta_t0)} --> {srt_ts(TOTAL_V)}\nMe siga para mais\n')
open(f'{P}/legendas.ass', 'w').write('\n'.join(ass) + '\n')
open(f'{P}/legendas.srt', 'w').write('\n'.join(srt))

# --- video: frame-exact trims, alternating framing, captions, logo ---
import os
os.makedirs(f'{P}/shots', exist_ok=True)
lst = []
for i, (a, b, zz) in enumerate(SHOTS):
    nfr = round(b * FPS) - round(a * FPS)
    crop = f'crop={cw}:{ch}:{cx}:{cy},' if zz == 'T' else ''
    out = f'{P}/shots/s{i:03d}.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{round(a * FPS) / FPS:.4f}', '-i', f'{P}/src.mov', '-an',
                    '-vf', f'{crop}scale={W}:{H}:flags=lanczos,setsar=1,fps={FPS}', '-frames:v', str(nfr),
                    '-c:v', 'libx264', '-crf', '10', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], check=True)
    lst.append(f"file '{out}'")
open(f'{P}/shots.txt', 'w').write('\n'.join(lst) + '\n')
vf = (f'fps={FPS},tpad=stop_mode=clone:stop={round(TAIL * FPS)},ass={P}/legendas.ass:fontsdir={E}/subfonts[sub];'
      f'[sub][1:v]overlay=x=(W-w)/2:y=170-h/2:format=auto,fade=t=in:st=0:d=0.3,'
      f'fade=t=out:st={TOTAL_V - 0.3:.3f}:d=0.3,format=yuv420p[vout]')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', f'{P}/shots.txt', '-i', f'{S}/proj/logo_ready.png',
                '-filter_complex', f'[0:v]{vf}', '-map', '[vout]', '-an',
                '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-r', str(FPS),
                f'{P}/video_noaudio.mp4'], check=True)

# --- audio ---
SR = 48000
src, _ = sf.read(f'{P}/a48.wav', dtype='float32')
parts = []
for a, b, _ in SHOTS:
    seg = src[round(a * FPS) * SR // FPS:round(b * FPS) * SR // FPS].copy()
    f = int(0.01 * SR)
    seg[:f] *= np.linspace(0, 1, f)[:, None]
    seg[-f:] *= np.linspace(1, 0, f)[:, None]
    parts.append(seg)
parts.append(np.zeros((round(TAIL * FPS) * SR // FPS, 2), np.float32))
sf.write(f'{P}/voice_raw.wav', np.concatenate(parts), SR)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{P}/voice_raw.wav', '-af',
                'highpass=f=80,acompressor=threshold=-18dB:ratio=3:attack=8:release=160:makeup=1',
                '-ar', str(SR), f'{P}/voice_proc.wav'], check=True)
voice, _ = sf.read(f'{P}/voice_proc.wav', dtype='float32')
meter = pyln.Meter(SR)
voice = pyln.normalize.loudness(voice, meter.integrated_loudness(voice), -16.0)
n = len(voice)
music, _ = sf.read(f'{P}/music.wav', dtype='float32')
music = pyln.normalize.loudness(music[:n], meter.integrated_loudness(music[:n]), -22.0)
mf = int(2.0 * SR)
music[-mf:] *= np.linspace(1, 0, mf)[:, None]
env = librosa.feature.rms(y=voice.mean(1), frame_length=960, hop_length=480)[0]
active = np.convolve((20 * np.log10(env + 1e-9) > -40).astype(float), np.ones(15) / 15, mode='same')
gain = 10 ** (np.interp(np.arange(n) / SR, np.arange(len(active)) * 0.01, -8.0 * np.clip(active * 1.5, 0, 1)) / 20)
mix = voice + music * gain[:, None]
for name, t_src, g in C['sfx']:
    x, _ = sf.read(f'{E}/sfx/{name}.wav', dtype='float32')
    i = int((to_out(t_src) - 0.14) * SR)
    j = min(n, i + len(x))
    mix[i:j] += x[:j - i] * 10 ** (g / 20)
f = int(0.3 * SR)
mix[:f] *= np.linspace(0, 1, f)[:, None]
mix[-f:] *= np.linspace(1, 0, f)[:, None]
mix = pyln.normalize.loudness(mix, meter.integrated_loudness(mix), -16.0)
peak = np.max(np.abs(mix))
if peak > 0.89:
    mix *= 0.89 / peak
sf.write(f'{P}/mix.wav', mix, SR)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{P}/video_noaudio.mp4', '-i', f'{P}/mix.wav',
                '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000', '-ac', '2',
                '-shortest', '-movflags', '+faststart', f'{P}/final.mp4'], check=True)
json.dump({'shots': SHOTS, 'total_v': TOTAL_V, 'cues': [(round(to_out(c[0]), 2), c[1]) for c in CUES]},
          open(f'{P}/edl.json', 'w'), ensure_ascii=False, indent=1)
print('shots', len(SHOTS), 'TOTAL_V', round(TOTAL_V, 2), 'LUFS', round(meter.integrated_loudness(mix), 2))
