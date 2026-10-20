import json, subprocess, sys, numpy as np, soundfile as sf, pyloudnorm as pyln, librosa
from PIL import ImageFont

S = sys.argv[1]
P, E = f'{S}/proj', f'{S}/env'
FPS = 30
W, H = 1080, 1920
SRC_W, SRC_H = 720, 1280

def fr(x):
    return round(x * FPS) / FPS

# (src_start, src_end, framing) — every boundary is a silence cut (0.08s padding kept)
SHOTS = [(2.50, 4.20, 'T'), (4.30, 6.50, 'W'), (6.80, 11.46, 'T'), (11.74, 12.61, 'W'),
         (12.71, 17.62, 'T'), (17.81, 21.58, 'W'), (21.86, 23.58, 'T'), (23.64, 27.09, 'W'),
         (27.34, 29.25, 'T'), (29.57, 30.85, 'W')]
SHOTS = [(fr(a), fr(b), z) for a, b, z in SHOTS]
ZOOM = 1.12
CROP_T = (round(SRC_W / ZOOM / 2) * 2, round(SRC_H / ZOOM / 2) * 2)
CROP_T_XY = (round((SRC_W - CROP_T[0]) / 2), 26)  # keeps the question card inside the frame

out_starts, acc = [], 0.0
for a, b, _ in SHOTS:
    out_starts.append(acc)
    acc += b - a
TOTAL = acc

def to_out(s):
    for (a, b, _), o in zip(SHOTS, out_starts):
        if s < a:
            return o
        if s <= b:
            return o + s - a
    return TOTAL

# --- captions (source times, from Whisper turbo phrase slices; text hand-corrected) ---
a16, sr16 = sf.read(f'{P}/a16.wav', dtype='float32')
db = 20 * np.log10(librosa.feature.rms(y=a16, frame_length=480, hop_length=160)[0] + 1e-9)
def dip_near(t, win=0.3):
    i0, i1 = int((t - win) * 100), int((t + win) * 100)
    return (i0 + int(np.argmin(db[i0:i1]))) / 100

split8 = dip_near(17.89 + (20.30 - 17.89) * 22 / 49)
CUES = [
    (2.58, 'A FALHA NÃO É OBRIGATÓRIA', ['FALHA', 'NÃO']),
    (4.38, 'APESAR DE MUITAS VEZES SER INTERESSANTE', []),
    (6.88, 'UMA, DUAS, ATÉ TRÊS REPETIÇÕES NA RESERVA', ['UMA,', 'DUAS,', 'ATÉ', 'TRÊS']),
    (9.64, 'OU SEJA, REPETIÇÕES QUE VOCÊ DEIXA GUARDADAS', ['GUARDADAS']),
    (11.82, 'ANTES DE FALHAR', []),
    (12.79, 'JÁ GERA UM ESTÍMULO', ['ESTÍMULO']),
    (14.51, 'SUFICIENTE PARA PROMOVER A HIPERTROFIA', ['HIPERTROFIA']),
    (17.89, 'PORÉM, O QUE ACONTECE', []),
    (split8, 'É QUE A MAIORIA DAS PESSOAS', ['MAIORIA']),
    (20.30, 'PRINCIPALMENTE OS INICIANTES', ['INICIANTES']),
    (21.94, 'NÃO SABEM TER ESSA NOÇÃO DE', []),
    (23.72, 'QUANTAS REPETIÇÕES ESTÃO NA RESERVA', ['RESERVA']),
    (25.66, 'E TREINAM MUITO LONGE DA FALHA', ['LONGE', 'DA', 'FALHA']),
    (27.42, 'E ACABAM DEIXANDO RESULTADOS', ['RESULTADOS']),
    (29.65, 'NA MESA', ['NA', 'MESA']),
]
FS = 96
font = ImageFont.truetype(f'{E}/subfonts/MontserratET.ttf', 100)
MAXW = 840  # safe zone (864) minus outline

def width(line, fs):
    # calibrated against libass output: glyphs ~0.64*fs, +1px spacing per char, outline both sides
    return font.getlength(line) * fs * 0.64 / 100 + len(line) + 16

def wrap(text):
    words = text.split()
    if width(text, FS) <= MAXW:
        return [words], FS
    best = None
    for k in range(1, len(words)):
        l1, l2 = ' '.join(words[:k]), ' '.join(words[k:])
        w = max(width(l1, FS), width(l2, FS))
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

HL, BASE = '&H00F0C686&', '&H00EBEBEB&'
ass = [
    '[Script Info]', 'ScriptType: v4.00+', f'PlayResX: {W}', f'PlayResY: {H}', 'WrapStyle: 2', 'ScaledBorderAndShadow: yes', '',
    '[V4+ Styles]',
    'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
    f'Style: Cap,Montserrat ET,{FS},{BASE},{BASE},&H00242424&,&H90000000&,0,0,0,0,100,100,1,0,1,8,4,5,108,108,0,1',
    '', '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
]
srt = []
for i, (s, text, hl) in enumerate(CUES):
    t0 = to_out(s)
    t1 = to_out(CUES[i + 1][0]) if i + 1 < len(CUES) else TOTAL
    lines = []
    wrapped, fs = wrap(text)
    for line in wrapped:
        lines.append(' '.join(f'{{\\c{HL}}}{w}{{\\c{BASE}}}' if w in hl else w for w in line))
    body = '\\N'.join(lines)
    ass.append(f'Dialogue: 0,{ts(t0)},{ts(t1)},Cap,,0,0,0,,{{\\an5\\pos(540,1498)\\fs{fs}\\fad(60,0)\\fscx90\\fscy90\\t(0,110,\\fscx100\\fscy100)}}{body}')
    srt.append(f'{i + 1}\n{srt_ts(t0)} --> {srt_ts(t1)}\n{text.capitalize()}\n')
open(f'{P}/legendas.ass', 'w').write('\n'.join(ass) + '\n')
open(f'{P}/legendas.srt', 'w').write('\n'.join(srt))

# --- video ---
fc = []
for i, (a, b, z) in enumerate(SHOTS):
    crop = f'crop={CROP_T[0]}:{CROP_T[1]}:{CROP_T_XY[0]}:{CROP_T_XY[1]},' if z == 'T' else ''
    fc.append(f'[0:v]trim=start_frame={round(a * FPS)}:end_frame={round(b * FPS)},setpts=PTS-STARTPTS,{crop}scale={W}:{H}:flags=lanczos,setsar=1[v{i}]')
fc.append(''.join(f'[v{i}]' for i in range(len(SHOTS))) + f'concat=n={len(SHOTS)}:v=1:a=0,fps={FPS}[cat]')
grade = ('eq=contrast=1.07:saturation=0.88:gamma=0.98,'
         'colorbalance=rs=-0.08:bs=0.09:rm=-0.06:gm=0.0:bm=0.07:rh=-0.03:bh=0.04,'
         "curves=all='0/0.02 0.25/0.22 0.75/0.79 1/0.98'")
fc.append(f"[cat]{grade},ass={P}/legendas.ass:fontsdir={E}/subfonts[sub]")
fc.append(f'[sub][1:v]overlay=x=(W-w)/2:y=170-h/2:format=auto,fade=t=in:st=0:d=0.3,fade=t=out:st={TOTAL - 0.3:.3f}:d=0.3,format=yuv420p[vout]')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{P}/src.mov', '-i', f'{P}/logo_ready.png',
                '-filter_complex', ';'.join(fc), '-map', '[vout]', '-an',
                '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-r', str(FPS),
                f'{P}/video_noaudio.mp4'], check=True)

# --- audio ---
SR = 48000
src, _ = sf.read(f'{P}/a48.wav', dtype='float32')
parts = []
for a, b, _ in SHOTS:
    seg = src[int(a * SR):int(b * SR)].copy()
    f = int(0.01 * SR)
    seg[:f] *= np.linspace(0, 1, f)[:, None]
    seg[-f:] *= np.linspace(1, 0, f)[:, None]
    parts.append(seg)
voice_raw = np.concatenate(parts)
sf.write(f'{P}/voice_raw.wav', voice_raw, SR)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{P}/voice_raw.wav', '-af',
                'highpass=f=80,acompressor=threshold=-18dB:ratio=3:attack=8:release=160:makeup=1',
                '-ar', str(SR), f'{P}/voice_proc.wav'], check=True)
voice, _ = sf.read(f'{P}/voice_proc.wav', dtype='float32')
meter = pyln.Meter(SR)
voice = pyln.normalize.loudness(voice, meter.integrated_loudness(voice), -16.0)
n = len(voice)

music, _ = sf.read(f'{E}/music.wav', dtype='float32')
music = music[:n]
music = pyln.normalize.loudness(music, meter.integrated_loudness(music), -22.0)
# ducking: -8 dB wherever the voice is active (20ms envelope, 150ms ramps)
env = librosa.feature.rms(y=voice.mean(1), frame_length=960, hop_length=480)[0]
active = (20 * np.log10(env + 1e-9) > -40).astype(float)
k = int(0.15 / 0.01)
active = np.convolve(active, np.ones(k) / k, mode='same')
gain_db = -8.0 * np.clip(active * 1.5, 0, 1)
gain = 10 ** (np.interp(np.arange(n) / SR, np.arange(len(gain_db)) * 0.01, gain_db) / 20)
mix = voice + music * gain[:, None]

def sfx(name, t_out, db_gain):
    x, _ = sf.read(f'{E}/sfx/{name}.wav', dtype='float32')
    i = int(t_out * SR)
    j = min(n, i + len(x))
    mix[i:j] += x[:j - i] * 10 ** (db_gain / 20)

SFX = [('pop', to_out(6.88), -16), ('impact', to_out(16.55), -14),
       ('whoosh', to_out(17.81) - 0.14, -14), ('impact', to_out(29.65), -13)]
for name, t, g in SFX:
    sfx(name, t, g)
f = int(0.3 * SR)
mix[:f] *= np.linspace(0, 1, f)[:, None]
mix[-f:] *= np.linspace(1, 0, f)[:, None]
mix = pyln.normalize.loudness(mix, meter.integrated_loudness(mix), -16.0)
peak = np.max(np.abs(mix))
if peak > 0.89:  # true-peak headroom ~ -1 dBTP
    mix *= 0.89 / peak
sf.write(f'{P}/mix.wav', mix, SR)

subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{P}/video_noaudio.mp4', '-i', f'{P}/mix.wav',
                '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000', '-ac', '2',
                '-shortest', '-movflags', '+faststart', f'{P}/final.mp4'], check=True)
json.dump({'shots': SHOTS, 'out_starts': out_starts, 'total': TOTAL, 'cues': [(to_out(c[0]), c[1]) for c in CUES],
           'sfx': SFX, 'split8': split8}, open(f'{P}/edl.json', 'w'), ensure_ascii=False, indent=1)
print('TOTAL', round(TOTAL, 2), 'LUFS', round(meter.integrated_loudness(mix), 2), 'peak', round(float(np.max(np.abs(mix))), 3))
