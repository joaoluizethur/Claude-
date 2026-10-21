# Original Bavarian-style polka (oompah) rendered with FluidR3 GM. usage: gen_polka.py out.mid
import sys, mido
TPB = 480; E = TPB // 2  # eighth
BPM = 120
mid = mido.MidiFile(ticks_per_beat=TPB)
def track(ch, prog, name):
    t = mido.MidiTrack(); mid.tracks.append(t)
    t.append(mido.MetaMessage('track_name', name=name))
    if ch != 9: t.append(mido.Message('program_change', channel=ch, program=prog, time=0))
    return t
meta = mido.MidiTrack(); mid.tracks.append(meta)
meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM)))
meta.append(mido.MetaMessage('time_signature', numerator=2, denominator=4))
Bb, F7, Eb = 'Bb', 'F7', 'Eb'
CH = {'Bb': ([58, 62, 65], 46, 53), 'F7': ([57, 60, 63], 41, 48), 'Eb': ([55, 58, 63], 51, 46)}  # voicing, root, fifth
A1 = [[(77,1),(74,1),(70,1),(74,1)], [(77,2),(77,1),(79,1)], [(75,1),(72,1),(69,1),(72,1)], [(75,2),(75,1),(77,1)],
      [(79,1),(77,1),(75,1),(72,1)], [(69,1),(72,1),(75,1),(79,1)], [(77,1),(74,1),(75,1),(72,1)], [(70,2),(None,2)]]
A2 = A1[:4] + [[(84,1),(81,1),(77,1),(75,1)], [(72,1),(74,1),(75,1),(77,1)], [(74,1),(77,1),(72,1),(75,1)], [(70,1),(65,1),(70,2)]]
CA = [Bb, Bb, F7, F7, F7, F7, Bb, Bb]
B = [[(79,2),(75,1),(79,1)], [(82,2),(79,2)], [(77,1),(74,1),(77,1),(82,1)], [(77,4)],
     [(75,1),(72,1),(75,1),(81,1)], [(77,2),(75,2)], [(74,1),(77,1),(75,1),(72,1)], [(70,2),(None,2)]]
CB = [Eb, Eb, Bb, Bb, F7, F7, Bb, Bb]
INTRO = [[(None,4)]] * 2
form = [(INTRO, [Bb, F7], 'none'), (A1, CA, 'clar'), (A2, CA, 'clar'), (A1, CA, 'tpt'), (A2, CA, 'tpt'),
        (B, CB, 'clar'), (B, CB, 'tpt'), (A1, CA, 'both'), (A2, CA, 'both')]
SCALE = [58, 60, 62, 63, 65, 67, 69]  # Bb major pitch classes base
def third_below(p):
    pcs = sorted({(x + 12 * k) for x in SCALE for k in range(-2, 4)})
    i = pcs.index(p) if p in pcs else min(range(len(pcs)), key=lambda j: abs(pcs[j] - p))
    return pcs[i - 2]
tuba, acc, clar, clar2, tpt, drum = track(0, 58, 'tuba'), track(1, 21, 'accordion'), track(2, 71, 'clarinet'), track(3, 71, 'clarinet2'), track(4, 56, 'trumpet'), track(9, 0, 'drums')
ev = {id(t): [] for t in (tuba, acc, clar, clar2, tpt, drum)}
def note(t, ch, p, start, dur, vel):
    ev[id(t)] += [(start, 1, mido.Message('note_on', channel=ch, note=p, velocity=vel)), (start + dur - 10, 0, mido.Message('note_off', channel=ch, note=p, velocity=0))]
bar_t = 4 * E; t0 = 0
for mel, chords, lead in form:
    for b, notes in enumerate(mel):
        bs = t0 + b * bar_t
        voicing, root, fifth = CH[chords[b % len(chords)]]
        note(tuba, 0, root, bs, 2 * E, 100); note(tuba, 0, fifth if fifth < root + 12 else fifth - 12, bs + 2 * E, 2 * E, 92)
        for off in (E, 3 * E):
            for p in voicing: note(acc, 1, p, bs + off, E - 40, 62)
        note(drum, 9, 36, bs, E, 90); note(drum, 9, 36, bs + 2 * E, E, 70)
        for off in (E, 3 * E): note(drum, 9, 42, bs + off, E // 2, 55)
        if b % 8 == 7: note(drum, 9, 49, bs, E, 50)
        pos = bs
        for p, d in notes:
            if p is not None and lead != 'none':
                if lead in ('clar', 'both'):
                    note(clar, 2, p, pos, d * E, 92); note(clar2, 3, third_below(p), pos, d * E, 74)
                if lead in ('tpt', 'both'):
                    note(tpt, 4, p - (12 if p > 80 else 0), pos, d * E, 88)
                    if lead == 'tpt': note(clar2, 3, third_below(p), pos, d * E, 66)
            pos += d * E
    t0 += len(mel) * bar_t
for t in (tuba, acc, clar, clar2, tpt, drum):
    last = 0
    for st, _, m in sorted(ev[id(t)], key=lambda x: (x[0], x[1])):
        t.append(m.copy(time=st - last)); last = st
mid.save(sys.argv[1]); print('bars', t0 // bar_t, 'seconds', t0 // bar_t * 4 * 60 / BPM / 2)
