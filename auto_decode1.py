cat > auto_decode1.py << 'EOF'
#!/usr/bin/env python3
"""
OOK Auto-Decoder — Termux/Android Edition
No compiler. No /tmp exec. Pure Python + numpy (if available).
Fast via: cumsum envelope, vectorised numpy ops, early-exit scoring.
"""
import array, os, sys, time, math, hashlib
from collections import Counter
from itertools import product

SAMPLE_RATE    = 32000
INPUT_FILE     = "filtered.raw"
REFERENCE_FILE = "stream3.bin"
OUTPUT_FILE    = "decoded_raw.bin"

# ── tuned to this signal: max=33, noise≈0-5, bursts≈16-33 ────────
THRESHOLDS = (
    [round(x*0.25,2) for x in range(2,21)]   # 0.5→5.0  fine
  + [round(x*0.5, 1) for x in range(10,41)]  # 5.0→20.0 medium
  + [round(x*1.0, 0) for x in range(20,35)]  # 20→34    coarse
)
PULSE_MS = [10,20,30,40,50,60,70,80,90,100,120,150]
GAP_MS   = [10,20,30,40,50,60,70,80,90,100,120,150]
WINDOWS_MS = [3,5,10,20]

R="\033[0m";BOLD="\033[1m";DIM="\033[2m"
CY="\033[36m";GR="\033[32m";YL="\033[33m"
RD="\033[31m";MG="\033[35m";BL="\033[34m"
UP="\033[A";EL="\033[2K"

def clr(n=1):
    for _ in range(n): print(UP+EL,end="")

def title(t,c=CY,w=62):
    p=(w-len(t)-2)//2
    print(f"{c}{BOLD}{'─'*p} {t} {'─'*(w-p-len(t)-2)}{R}")

def pbar(done,total,width=52,c=GR):
    f=int(width*done/total) if total else 0
    pct=100*done/total if total else 0
    return f"{c}{'█'*f}{DIM}{'░'*(width-f)}{R} {BOLD}{pct:5.1f}%{R}"

def hbar(v,mx,width=52,c=GR):
    f=int(width*v/mx) if mx else 0
    return f"{c}{'█'*f}{DIM}{'░'*(width-f)}{R}"

def fmt_t(s):
    s=int(s)
    if s<60:   return f"{s}s"
    if s<3600: return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"

SPARKS=" ▁▂▃▄▅▆▇█"
def spark(vals,width=60):
    if not vals: return "─"*width
    lo,hi=min(vals),max(vals);rng=hi-lo
    if rng==0: return ("▄" if lo>0 else "─")*width
    step=max(1,len(vals)//width)
    b=[vals[i] for i in range(0,len(vals),step)][:width]
    return "".join(SPARKS[min(8,int(9*(v-lo)/rng))] for v in b)

# ── check numpy ───────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NP=True
except ImportError:
    HAS_NP=False

def np_hint():
    if not HAS_NP:
        print(f"  {YL}tip: pip install numpy --break-system-packages  "
              f"→ 10× faster{R}")

# ── load ──────────────────────────────────────────────────────────
def load_raw():
    title("LOAD  RAW SIGNAL")
    sz=os.path.getsize(INPUT_FILE)
    t0=time.time()
    with open(INPUT_FILE,'rb') as f: raw=f.read()
    n=len(raw)//2
    if HAS_NP:
        samp=np.frombuffer(raw[:n*2],dtype=np.int16)
        print(f"  {GR}✓{R}  numpy  {sz/1024:.0f}KB  {n:,} samples  "
              f"{n/SAMPLE_RATE:.2f}s  [{time.time()-t0:.2f}s]\n")
    else:
        samp=array.array('h',raw[:n*2])
        print(f"  {GR}✓{R}  stdlib  {sz/1024:.0f}KB  {n:,} samples  "
              f"{n/SAMPLE_RATE:.2f}s  [{time.time()-t0:.2f}s]\n")
    np_hint()
    return samp,n

# ── envelope builders ─────────────────────────────────────────────
def envelope_np(samp,half):
    """Numpy cumsum envelope — O(n), fast."""
    absv=np.abs(samp).astype(np.float32)
    cs=np.zeros(len(absv)+1,dtype=np.float64)
    np.cumsum(absv,out=cs[1:])
    i=np.arange(len(absv))
    lo=np.maximum(0,i-half)
    hi=np.minimum(len(absv),i+half)
    return ((cs[hi]-cs[lo])/(hi-lo)).astype(np.float32)

def envelope_py(samp,half):
    """Pure Python cumsum envelope — O(n)."""
    n=len(samp)
    absv=[abs(s) for s in samp]
    cs=[0]*(n+1)
    for i,v in enumerate(absv): cs[i+1]=cs[i]+v
    env=[0.0]*n
    for i in range(n):
        lo=max(0,i-half); hi=min(n,i+half)
        env[i]=(cs[hi]-cs[lo])/(hi-lo)
    return env

def make_envelope(samp,window_ms):
    half=max(1,int(window_ms*SAMPLE_RATE/2000))
    if HAS_NP: return envelope_np(samp,half)
    return envelope_py(samp,half)

# ── decode ────────────────────────────────────────────────────────
def decode_np(env,threshold,min_pulse,min_gap):
    """Vectorised numpy OOK decode."""
    binary=(env>threshold).astype(np.int8)
    # find rising/falling edges
    diff=np.diff(binary,prepend=0)
    starts=np.where(diff==1)[0]
    ends  =np.where(diff==-1)[0]
    # align starts/ends
    if len(ends)==0 or len(starts)==0: return []
    if ends[0]<starts[0]: ends=ends[1:]
    mn=min(len(starts),len(ends))
    starts=starts[:mn]; ends=ends[:mn]
    widths=ends-starts
    # filter by min pulse
    mask=widths>=min_pulse
    starts=starts[mask]; ends=ends[mask]
    if len(starts)==0: return []
    # compute gaps between consecutive pulses
    bits=[1]
    for i in range(1,len(starts)):
        gap=starts[i]-ends[i-1]
        if gap>=min_gap: bits.append(0)
        bits.append(1)
    return bits

def decode_py(env,threshold,min_pulse,min_gap):
    """Pure Python OOK decode."""
    n=len(env); bits=[]; last=0; i=0
    while i<n:
        if env[i]>threshold:
            j=i
            while j<n and env[j]>threshold: j+=1
            if j-i>=min_pulse:
                if last and (i-last)>=min_gap: bits.append(0)
                bits.append(1); last=j
            i=j
        else: i+=1
    return bits

def decode(env,threshold,pulse_ms,gap_ms):
    mp=max(1,int(pulse_ms*SAMPLE_RATE/1000))
    mg=max(1,int(gap_ms  *SAMPLE_RATE/1000))
    if HAS_NP and isinstance(env,np.ndarray):
        return decode_np(env,threshold,mp,mg)
    return decode_py(env,threshold,mp,mg)

# ── scoring ───────────────────────────────────────────────────────
def score_bits(bits):
    nb=len(bits)
    if nb<16: return 0.0

    # 1. balance
    ones=sum(bits); r=ones/nb
    if r<0.03 or r>0.97: return 0.0
    bal=1-2*abs(r-0.5)

    # 2. run-length (ideal≈8 for byte protocols)
    run=1; runs=0; rsum=0
    for i in range(1,nb):
        if bits[i]==bits[i-1]: run+=1
        else: rsum+=run; runs+=1; run=1
    avg_run=rsum/runs if runs else 1
    run_sc=1/(1+abs(avg_run-8))

    # 3. periodicity (sample up to 512 periods for speed)
    period_sc=0.0
    limit=min(nb//2, 512)
    for period in range(8,limit,max(1,limit//64)):
        matches=sum(bits[i]==bits[i+period] for i in range(nb-period))
        q=matches/(nb-period)
        if q>period_sc: period_sc=q
        if q>0.92: break

    # 4. byte entropy bonus
    byte_sc=0.0
    if nb>=64:
        nbytes=nb//8
        freq=Counter(
            sum(bits[i*8+j]<<(7-j) for j in range(8))
            for i in range(nbytes)
        )
        ent=0.0
        for c in freq.values():
            p=c/nbytes; ent-=p*math.log2(p)
        byte_sc=max(0, 1-(ent/8.0))

    return nb*(0.25*bal + 0.25*run_sc + 0.30*period_sc + 0.20*byte_sc)

# ── preview ───────────────────────────────────────────────────────
def preview(samp,n):
    title("SIGNAL PREVIEW  (5ms envelope)")
    t0=time.time()
    env=make_envelope(samp,5)
    step=max(1,n//80)
    if HAS_NP: snaps=env[::step][:80].tolist()
    else:      snaps=[env[i] for i in range(0,n,step)][:80]
    mid=len(snaps)//2
    print(f"  {CY}0→{n//2//SAMPLE_RATE:.0f}s :{R} {GR}{spark(snaps[:mid],mid)}{R}")
    print(f"  {CY}  →{n//SAMPLE_RATE:.0f}s :{R} {GR}{spark(snaps[mid:],len(snaps)-mid)}{R}")
    emax=max(snaps); emean=sum(snaps)/len(snaps)
    print(f"  max={emax:.2f}  mean={emean:.2f}  [{time.time()-t0:.2f}s]\n")

# ── sweep ─────────────────────────────────────────────────────────
def sweep(samp,n):
    title("SWEEP  —  EVERY COMBO  SCORED")

    combos=[]
    for wms in WINDOWS_MS:
        for th in THRESHOLDS:
            for p in PULSE_MS:
                for g in GAP_MS:
                    combos.append((wms,th,p,g))
    total=len(combos)

    mode="numpy" if HAS_NP else "python"
    print(f"  {BOLD}{total:,}{R} combos  "
          f"({len(WINDOWS_MS)}w × {len(THRESHOLDS)}th × "
          f"{len(PULSE_MS)}p × {len(GAP_MS)}g)  [{mode}]\n")

    # pre-build all envelopes (one per window size)
    title("  PRE-BUILDING ENVELOPES",CY)
    envs={}
    for wms in WINDOWS_MS:
        t0=time.time()
        print(f"  window {wms:>3}ms ...", end='\r', flush=True)
        envs[wms]=make_envelope(samp,wms)
        print(f"  window {wms:>3}ms  {GR}✓{R}  [{time.time()-t0:.2f}s]")
    print()

    best_score=-1.0
    best_params=None
    best_bits=[]
    sc_h=[]; bit_h=[]
    t0=time.time(); last_draw=0.0
    last_wms=None

    for idx,(wms,th,p,g) in enumerate(combos,1):
        env=envs[wms]
        bits=decode(env,th,p,g)
        sc=score_bits(bits)
        sc_h.append(sc); bit_h.append(len(bits))

        if sc>best_score:
            best_score=sc
            best_params=(wms,th,p,g)
            best_bits=bits

        now=time.time()
        if now-last_draw >= 0.08:
            last_draw=now
            el=now-t0; rate=idx/el if el else 0; eta=(total-idx)/rate if rate else 0
            clr(9)
            print(f"  {CY}score  :{R} {YL}{spark(sc_h[-60:],60)}{R}  "
                  f"best={BOLD}{best_score:.1f}{R}")
            print(f"  {CY}bits   :{R} {MG}{spark(bit_h[-60:],60)}{R}  "
                  f"peak={BOLD}{max(bit_h)}{R}")
            print(f"  {pbar(idx,total,52,BL)}")
            print(f"  {fmt_t(el)} elapsed  ETA {fmt_t(eta)}  "
                  f"{BOLD}{rate:.0f}{R} combos/s")
            if best_params:
                bw,bt,bp,bg=best_params
                print(f"  {CY}best →{R} "
                      f"win={BOLD}{bw}ms{R}  "
                      f"th={BOLD}{bt:.2f}{R}  "
                      f"pulse={BOLD}{bp}ms{R}  "
                      f"gap={BOLD}{bg}ms{R}")
            else:
                print(f"  {DIM}searching...{R}")
            print(f"  {CY}now  →{R} "
                  f"win={BOLD}{wms}ms{R}  "
                  f"th={BOLD}{th:.2f}{R}  "
                  f"score={BOLD}{YL}{sc:.1f}{R}  "
                  f"bits={BOLD}{len(bits)}{R}")
            # bit density strip
            pk=max(bit_h) if bit_h else 1
            density=''.join(
                f"{GR}█{R}" if b>50 else
                f"{YL}▄{R}" if b>10 else
                f"{DIM}·{R}"
                for b in bit_h[-60:]
            )
            print(f"  {CY}density:{R} {density}")
            # current best bit strip
            if best_bits:
                strip=''.join(
                    f"{GR}█{R}" if b else f"{DIM}░{R}"
                    for b in best_bits[:60]
                )
                print(f"  {CY}bits   :{R} {strip}",flush=True)
            else:
                print(f"  {DIM}{'─'*60}{R}",flush=True)

    el=time.time()-t0; clr(9)
    # final display
    print(f"  {CY}score  :{R} {YL}{spark(sc_h[-60:],60)}{R}  "
          f"best={BOLD}{best_score:.1f}{R}")
    print(f"  {CY}bits   :{R} {MG}{spark(bit_h[-60:],60)}{R}  "
          f"peak={BOLD}{max(bit_h) if bit_h else 0}{R}")
    print(f"\n  {GR}{BOLD}✅  SWEEP COMPLETE{R}  "
          f"[{el:.1f}s]  {total/el:.0f} combos/s")

    if best_params:
        bw,bt,bp,bg=best_params
        print(f"\n  {CY}window   {R} {BOLD}{bw}ms{R}")
        print(f"  {CY}threshold{R} {BOLD}{bt:.2f}{R}")
        print(f"  {CY}pulse    {R} {BOLD}{bp}ms{R}")
        print(f"  {CY}gap      {R} {BOLD}{bg}ms{R}")
        print(f"  {CY}score    {R} {BOLD}{YL}{best_score:.2f}{R}")
        print(f"  {CY}bits     {R} {BOLD}{len(best_bits)}{R}\n")
    else:
        print(f"  {RD}no bits found{R}\n")

    return best_bits, best_params, best_score

# ── output ────────────────────────────────────────────────────────
def bits_to_bytes(bits):
    while len(bits)%8: bits.append(0)
    out=bytearray()
    for i in range(0,len(bits),8):
        b=0
        for j in range(8): b|=(bits[i+j]<<(7-j))
        out.append(b)
    return out

def hexdump(data,rows=8):
    title("HEXDUMP  (first 128 bytes)",GR)
    for r in range(rows):
        row=data[r*16:(r+1)*16]
        if not row: break
        h=' '.join(f'{b:02x}' for b in row)
        c=''.join(chr(b) if 32<=b<127 else '·' for b in row)
        print(f"  {DIM}{r*16:04x}{R}  {CY}{h:<48}{R}  {YL}{c}{R}")

def analyse(data):
    title("BYTE ANALYSIS",MG)
    n=len(data)
    if not n: return
    freq=Counter(data)
    ent=0.0
    for c in freq.values():
        p=c/n; ent-=p*math.log2(p)
    structure='high — compressed/encrypted' if ent>7 else \
              'moderate' if ent>5 else \
              'low — structured protocol data'
    print(f"  {n} bytes  |  entropy {ent:.3f} bits/byte  ({structure})")
    print(f"  top bytes: "+
          "  ".join(f"{BOLD}0x{b:02x}{R}×{c}" for b,c in freq.most_common(6)))

    # ASCII strings
    runs=[]; cur=[]
    for byte in data:
        if 32<=byte<127: cur.append(chr(byte))
        else:
            if len(cur)>=4: runs.append(''.join(cur))
            cur=[]
    if len(cur)>=4: runs.append(''.join(cur))
    if runs:
        print(f"  {GR}ASCII strings:{R}")
        for s in runs[:8]: print(f"    \"{YL}{s}{R}\"")

    # repeating block
    for blen in [4,8,16,32]:
        if n<blen*3: continue
        ref=bytes(data[:blen])
        hits=sum(1 for i in range(0,n-blen,blen) if bytes(data[i:i+blen])==ref)
        if hits>2:
            print(f"  {GR}repeating {blen}B block ×{hits}:{R} "
                  f"{' '.join(f'{b:02x}' for b in ref)}")
            break
    print()

def compare_ref(decoded):
    if not os.path.exists(REFERENCE_FILE):
        print(f"  {DIM}{REFERENCE_FILE} not present — skipping{R}\n"); return
    title("VS  REFERENCE",MG)
    ref=open(REFERENCE_FILE,'rb').read()
    dl,rl=len(decoded),len(ref)
    print(f"  decoded={dl}B  reference={rl}B")
    if dl==rl:
        m=sum(a==b for a,b in zip(decoded,ref)); pct=100*m/rl
        c=GR if pct>80 else YL if pct>50 else RD
        print(f"  {hbar(m,rl,50,c)}  {c}{BOLD}{pct:.1f}%{R}  ({m}/{rl})")
        if pct==100: print(f"  {GR}{BOLD}PERFECT MATCH ✓{R}")
    elif dl<rl:
        if   ref.startswith(bytes(decoded)): print(f"  {GR}✓ matches START{R}")
        elif ref.endswith(bytes(decoded)):   print(f"  {GR}✓ matches END{R}")
        else:
            best_m=0; best_off=0
            for off in range(0,rl-dl,max(1,(rl-dl)//500)):
                m=sum(a==b for a,b in zip(decoded,ref[off:off+dl]))
                if m>best_m: best_m=m; best_off=off
            pct=100*best_m/dl if dl else 0
            c=GR if pct>80 else YL if pct>50 else RD
            print(f"  best offset {best_off}: {c}{BOLD}{pct:.1f}%{R} match")
    else:
        print(f"  {RD}decoded longer than reference{R}")
    print()

# ── main ──────────────────────────────────────────────────────────
def main():
    os.system("clear")
    runtime="numpy "+__import__('numpy').version.version if HAS_NP else "pure Python"
    print(f"\n{CY}{BOLD}"
          f"╔══════════════════════════════════════════════════════════════╗\n"
          f"║  OOK AUTO-DECODER  ·  TERMUX EDITION  ·  {runtime:<20} ║\n"
          f"╚══════════════════════════════════════════════════════════════╝"
          f"{R}\n")

    if not os.path.exists(INPUT_FILE):
        print(f"  {RD}❌  {INPUT_FILE} not found{R}\n"
              f"  Run: ffmpeg -i high_15k.wav \\\n"
              f"    -af \"bandpass=f=15500:width_type=h:width=2000,volume=200\" \\\n"
              f"    -f s16le -acodec pcm_s16le filtered.raw")
        sys.exit(1)

    samp,n        = load_raw()
    preview(samp,n)
    bits,params,sc = sweep(samp,n)

    if not bits or not params:
        print(f"  {RD}⚠  No signal decoded.{R}")
        print(f"  {YL}Run python3 diagnose.py — then try more gain:{R}")
        print(f"  ffmpeg ... volume=500 ... filtered.raw")
        sys.exit(1)

    title("OUTPUT",YL)
    bd=bits_to_bytes(bits[:])
    with open(OUTPUT_FILE,'wb') as f: f.write(bd)
    chk=hashlib.md5(bd).hexdigest()
    print(f"  {GR}✓{R}  {len(bits)} bits  →  "
          f"{len(bd)} bytes  →  {BOLD}{OUTPUT_FILE}{R}")
    print(f"  md5  {DIM}{chk}{R}\n")

    hexdump(bd)
    print()
    analyse(bd)
    compare_ref(bd)
    print(f"{GR}{BOLD}  ✅  Mission complete.{R}\n")

if __name__ == "__main__":
    main()
EOF
python3 auto_decode1.py
