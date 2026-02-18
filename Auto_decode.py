'''bash

cat > auto_decode.py << 'EOF'
#!/usr/bin/env python3
"""
OOK Auto-Decoder — tuned to this specific signal
envelope max=33, signal bursts at ~16-33, noise floor ~0-5
"""
import array, os, sys, time, ctypes, tempfile, subprocess
from itertools import product

SAMPLE_RATE    = 32000
INPUT_FILE     = "filtered.raw"
REFERENCE_FILE = "stream3.bin"
OUTPUT_FILE    = "decoded_raw.bin"

# ── TUNED TO THIS SIGNAL ──────────────────────────────────────────
# Noise floor ~0-5, signal bursts visible at 16-33
# Sweep the gap between noise and signal peak
THRESHOLDS = [0.5,1,1.5,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,25,28,30]
# Pulse widths: diagnose showed avg_on ~68-130ms, sweep around that
PULSE_MS   = [10,20,30,40,50,60,70,80,90,100,120,150]
# Gap widths: similar range
GAP_MS     = [10,20,30,40,50,60,70,80,90,100,120,150]
# 26×12×12 = 3744 combos

R="\033[0m";BOLD="\033[1m";DIM="\033[2m"
CY="\033[36m";GR="\033[32m";YL="\033[33m"
RD="\033[31m";MG="\033[35m";BL="\033[34m"
UP="\033[A";EL="\033[2K"

def clr(n=1):
    for _ in range(n): print(UP+EL,end="")

def title(t,c=CY):
    w=58;p=(w-len(t)-2)//2
    print(f"{c}{BOLD}{'─'*p} {t} {'─'*(w-p-len(t)-2)}{R}")

def pbar(done,total,width=44,c=GR):
    f=int(width*done/total) if total else 0
    pct=100*done/total if total else 0
    return f"{c}{'█'*f}{DIM}{'░'*(width-f)}{R} {BOLD}{pct:5.1f}%{R}"

def hbar(v,mx,width=44,c=GR):
    f=int(width*v/mx) if mx else 0
    return f"{c}{'█'*f}{DIM}{'░'*(width-f)}{R}"

def fmt_t(s):
    s=int(s)
    if s<60:   return f"{s}s"
    if s<3600: return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"

SPARK=" ▁▂▃▄▅▆▇█"
def spark(vals,width=54):
    if not vals: return "─"*width
    lo,hi=min(vals),max(vals);rng=hi-lo
    if rng==0: return "─"*width
    step=max(1,len(vals)//width)
    b=[vals[i] for i in range(0,len(vals),step)][:width]
    return "".join(SPARK[min(8,int(9*(v-lo)/rng))] for v in b)

# ── C source ──────────────────────────────────────────────────────
C_SRC=r"""
#include <math.h>
#include <stdlib.h>

int decode_ook(const float*env,int n,float th,int minp,int ming,int*out){
    int nb=0,last=0,i=0;
    while(i<n){
        if(env[i]>th){
            int j=i;
            while(j<n&&env[j]>th)j++;
            if(j-i>=minp){
                if(last&&(i-last)>=ming)out[nb++]=0;
                out[nb++]=1;last=j;
            }
            i=j;
        }else i++;
    }
    return nb;
}

float score(const int*bits,int nb){
    if(nb<16)return 0;
    int ones=0;
    for(int i=0;i<nb;i++)ones+=bits[i];
    float r=(float)ones/nb;
    if(r<0.05f||r>0.95f)return 0;
    float bal=1-2*fabsf(r-0.5f);
    /* run length analysis */
    int run=1,runs=0,rsum=0;
    for(int i=1;i<nb;i++){
        if(bits[i]==bits[i-1])run++;
        else{rsum+=run;runs++;run=1;}
    }
    float avg=runs?(float)rsum/runs:1;
    /* bonus: repetition/periodicity — check if bit pattern repeats */
    int rep_bonus=0;
    for(int period=8;period<=nb/3;period+=8){
        int matches=0;
        for(int i=0;i<nb-period;i++)
            if(bits[i]==bits[i+period])matches++;
        if((float)matches/(nb-period)>0.85f){rep_bonus=period;break;}
    }
    float rep=rep_bonus?(float)rep_bonus/nb*10:0;
    return (float)nb*(0.35f*bal + 0.35f/(1+fabsf(avg-8)) + 0.30f*rep/(1+rep));
}

float sweep(const float*env,int n,
            const float*ths,int nt,
            const int*ps,int np,
            const int*gs,int ng,
            float*bth,int*bp,int*bg,int*bnb,
            void(*cb)(int,int,float,float,int,int,int)){
    int*buf=(int*)malloc(n*sizeof(int));
    int total=nt*np*ng,done=0;
    float best=-1;*bnb=0;
    for(int ti=0;ti<nt;ti++)
    for(int pi=0;pi<np;pi++)
    for(int gi=0;gi<ng;gi++){
        int nb=decode_ook(env,n,ths[ti],ps[pi],gs[gi],buf);
        float sc=score(buf,nb);
        if(sc>best){best=sc;*bth=ths[ti];*bp=ps[pi];*bg=gs[gi];*bnb=nb;}
        done++;
        if(cb&&done%32==0)cb(done,total,sc,ths[ti],ps[pi],gs[gi],nb);
    }
    free(buf);
    return best;
}
"""

def compile_c():
    title("COMPILE  C DECODER",MG)
    src=tempfile.NamedTemporaryFile(suffix='.c',delete=False,mode='w')
    src.write(C_SRC); src.close()
    so=src.name.replace('.c','.so')
    for cc in ['cc','gcc','clang']:
        cmd=[cc,'-O3','-march=native','-ffast-math','-shared',
             '-fPIC','-o',so,src.name,'-lm']
        print(f"  {DIM}{' '.join(cmd)}{R}")
        r=subprocess.run(cmd,capture_output=True,text=True)
        if r.returncode==0:
            print(f"  {GR}✓  compiled [{os.path.getsize(so)/1024:.1f}KB]{R}\n")
            lib=ctypes.CDLL(so)
            FLT=ctypes.c_float;INT=ctypes.c_int;PF=ctypes.POINTER(FLT);PI=ctypes.POINTER(INT)
            lib.decode_ook.restype=INT
            lib.decode_ook.argtypes=[PF,INT,FLT,INT,INT,PI]
            lib.score.restype=FLT
            lib.score.argtypes=[PI,INT]
            lib.sweep.restype=FLT
            lib.sweep.argtypes=[PF,INT,PF,INT,PI,INT,PI,INT,PF,PI,PI,PI,ctypes.c_void_p]
            return lib,so
        print(f"  {YL}{cc} failed{R}")
    print(f"  {RD}no compiler — pure Python fallback{R}\n")
    return None,None

def load_and_envelope():
    title("LOAD + ENVELOPE")
    with open(INPUT_FILE,'rb') as f: raw=f.read()
    n=len(raw)//2
    samp=array.array('h',raw[:n*2])
    print(f"  {n:,} samples / {n/SAMPLE_RATE:.1f}s",flush=True)

    half=int(0.005*SAMPLE_RATE)//2
    absv=[abs(s) for s in samp]
    cs=[0]*(n+1)
    for i,v in enumerate(absv): cs[i+1]=cs[i]+v

    env=(ctypes.c_float*n)()
    snaps=[];snap_step=max(1,n//80);rep=max(1,n//30)
    t0=time.time()
    for i in range(n):
        lo=max(0,i-half);hi=min(n,i+half)
        v=(cs[hi]-cs[lo])/(hi-lo)
        env[i]=v
        if i%snap_step==0: snaps.append(v)
        if i%rep==0 and i>0:
            el=time.time()-t0;eta=el*(n-i)/i
            clr(2)
            print(f"  {CY}{spark(snaps[-80:],80)}{R}")
            print(f"  {pbar(i,n,44,MG)}  ETA {fmt_t(eta)}",flush=True)

    clr(2)
    vals=list(env)
    emin,emax,emean=min(vals),max(vals),sum(vals)/n
    print(f"  {CY}{spark(snaps,80)}{R}")
    print(f"  {GR}✓{R}  [{time.time()-t0:.1f}s]  "
          f"min={emin:.1f} mean={emean:.1f} max={emax:.1f}\n")
    return env,n,emin,emax,emean

def sweep_c(lib,env,n):
    title("SWEEP  (C/O3)  —  scoring ALL combos")
    total=len(THRESHOLDS)*len(PULSE_MS)*len(GAP_MS)
    print(f"  {total:,} combos  |  best-score wins  |  periodicity bonus\n")

    FLT=ctypes.c_float;INT=ctypes.c_int
    th_a=(FLT*len(THRESHOLDS))(*[float(t) for t in THRESHOLDS])
    p_a =(INT*len(PULSE_MS))  (*[int(p*SAMPLE_RATE//1000) for p in PULSE_MS])
    g_a =(INT*len(GAP_MS))    (*[int(g*SAMPLE_RATE//1000) for g in GAP_MS])

    sc_h=[];bit_h=[];t0=time.time();last=[0.0]

    CB=ctypes.CFUNCTYPE(None,INT,INT,FLT,FLT,INT,INT,INT)
    def _cb(done,tot,sc,th,p,g,nb):
        sc_h.append(float(sc));bit_h.append(nb)
        now=time.time()
        if now-last[0]<0.07: return
        last[0]=now
        el=now-t0;rate=done/el if el else 0;eta=(tot-done)/rate if rate else 0
        pm=p*1000//SAMPLE_RATE;gm=g*1000//SAMPLE_RATE
        clr(7)
        print(f"  {CY}score :{R} {YL}{spark(sc_h[-80:],80)}{R}  pk={max(sc_h):.0f}")
        print(f"  {CY}bits  :{R} {MG}{spark(bit_h[-80:],80)}{R}  pk={max(bit_h)}")
        print(f"  {pbar(done,tot,54,BL)}")
        print(f"  {fmt_t(el)} elapsed  ETA {fmt_t(eta)}  {rate:.0f} combos/s")
        print(f"  {CY}th{R} {BOLD}{th:>6.1f}{R}  "
              f"{CY}pulse{R} {BOLD}{pm:>4}ms{R}  "
              f"{CY}gap{R} {BOLD}{gm:>4}ms{R}")
        print(f"  {CY}score{R} {BOLD}{YL}{sc:>10.1f}{R}  "
              f"{CY}bits{R} {BOLD}{nb:>6}{R}")
        # bit density meter
        density=nb/total if total else 0
        print(f"  {CY}density{R} {hbar(min(nb,500),500,40,GR)}",flush=True)

    cb=CB(_cb)
    bth=FLT(0);bp=INT(0);bg=INT(0);bnb=INT(0)
    best=lib.sweep(env,n,th_a,len(THRESHOLDS),p_a,len(PULSE_MS),
                   g_a,len(GAP_MS),
                   ctypes.byref(bth),ctypes.byref(bp),
                   ctypes.byref(bg),ctypes.byref(bnb),cb)

    el=time.time()-t0; clr(7)
    print(f"  {CY}score :{R} {YL}{spark(sc_h[-80:],80)}{R}  pk={max(sc_h) if sc_h else 0:.0f}")
    print(f"  {CY}bits  :{R} {MG}{spark(bit_h[-80:],80)}{R}  pk={max(bit_h) if bit_h else 0}")
    print(f"\n  {GR}{BOLD}✅  SWEEP DONE{R}  [{el:.2f}s]  {total/el:.0f} combos/s")
    print(f"  best score {BOLD}{YL}{best:.1f}{R}")
    pm=bth.value; pp=bp.value*1000//SAMPLE_RATE; pg=bg.value*1000//SAMPLE_RATE
    print(f"  threshold {BOLD}{pm:.1f}{R}  pulse {BOLD}{pp}ms{R}  gap {BOLD}{pg}ms{R}")
    print(f"  bits {BOLD}{bnb.value}{R}\n")
    return bth.value,bp.value,bg.value,bnb.value

def decode_py(env,threshold,p_samp,g_samp):
    n=len(env);bits=[];last=0;i=0
    while i<n:
        if env[i]>threshold:
            j=i
            while j<n and env[j]>threshold: j+=1
            if j-i>=p_samp:
                if last and (i-last)>=g_samp: bits.append(0)
                bits.append(1);last=j
            i=j
        else: i+=1
    return bits

def sweep_py(env_list,n):
    title("SWEEP  (Python fallback)")
    combos=list(product(THRESHOLDS,PULSE_MS,GAP_MS));total=len(combos)
    print(f"  {YL}⚠  no C compiler — Python mode (slow){R}\n")

    def score(bits):
        nb=len(bits)
        if nb<16: return 0
        ones=sum(bits);r=ones/nb
        if r<0.05 or r>0.95: return 0
        bal=1-2*abs(r-0.5)
        run=1;runs=0;rs=0
        for i in range(1,nb):
            if bits[i]==bits[i-1]: run+=1
            else: rs+=run;runs+=1;run=1
        avg=rs/runs if runs else 1
        return nb*(0.5*bal+0.5/(1+abs(avg-8)))

    best=-1;bp=None;bb=None;sch=[];bith=[];t0=time.time()
    for idx,(th,p,g) in enumerate(combos,1):
        bits=decode_py(env_list,th,int(p*SAMPLE_RATE/1000),int(g*SAMPLE_RATE/1000))
        sc=score(bits);sch.append(sc);bith.append(len(bits))
        if sc>best: best=sc;bp=(th,p,g);bb=bits
        if idx%30==0:
            el=time.time()-t0;rate=idx/el if el else 0;eta=(total-idx)/rate if rate else 0
            clr(6)
            print(f"  {CY}score:{R} {YL}{spark(sch[-80:],80)}{R}  pk={max(sch):.0f}")
            print(f"  {CY}bits :{R} {MG}{spark(bith[-80:],80)}{R}  pk={max(bith)}")
            print(f"  {pbar(idx,total,54,BL)}")
            print(f"  {fmt_t(el)} elapsed  ETA {fmt_t(eta)}  {rate:.0f}/s")
            print(f"  th={bp[0]}  p={bp[1]}ms  g={bp[2]}ms  "
                  f"score={BOLD}{YL}{best:.1f}{R}  bits={len(bb)}",flush=True)
    el=time.time()-t0;clr(6)
    print(f"  {GR}✓{R}  [{el:.1f}s]  score={best:.1f}  "
          f"th={bp[0]} p={bp[1]}ms g={bp[2]}ms  bits={len(bb)}\n")
    return float(bp[0]),int(bp[1]*SAMPLE_RATE//1000),int(bp[2]*SAMPLE_RATE//1000),len(bb)

def bits_to_bytes(bits):
    while len(bits)%8: bits.append(0)
    out=bytearray()
    for i in range(0,len(bits),8):
        b=0
        for j in range(8): b|=(bits[i+j]<<(7-j))
        out.append(b)
    return out

def hexdump(data,rows=8):
    title("HEXDUMP",GR)
    for r in range(rows):
        row=data[r*16:(r+1)*16]
        if not row: break
        h=' '.join(f'{b:02x}' for b in row)
        c=''.join(chr(b) if 32<=b<127 else '.' for b in row)
        print(f"  {DIM}{r*16:04x}{R}  {CY}{h:<48}{R}  {YL}{c}{R}")

def compare_ref(decoded):
    if not os.path.exists(REFERENCE_FILE):
        print(f"  {DIM}no {REFERENCE_FILE}{R}\n"); return
    title("VS REFERENCE",MG)
    ref=open(REFERENCE_FILE,'rb').read()
    dl,rl=len(decoded),len(ref)
    print(f"  decoded={dl}B  reference={rl}B")
    if dl==rl:
        m=sum(a==b for a,b in zip(decoded,ref));pct=100*m/rl
        c=GR if pct>80 else YL if pct>50 else RD
        print(f"  {hbar(m,rl,50,c)}  {c}{BOLD}{pct:.1f}%{R}  ({m}/{rl})\n")
    elif dl<rl:
        if ref.startswith(decoded):  print(f"  {GR}✓ matches START{R}\n")
        elif ref.endswith(decoded):  print(f"  {GR}✓ matches END{R}\n")
        else:                        print(f"  {YL}no prefix/suffix match{R}\n")
    else: print(f"  {RD}decoded longer than reference{R}\n")

def main():
    os.system("clear")
    print(f"\n{CY}{BOLD}╔══════════════════════════════════════════════════════╗")
    print(f"║     OOK AUTO-DECODER  ·  PRO EDITION  ·  C+ctypes    ║")
    print(f"╚══════════════════════════════════════════════════════╝{R}\n")
    if not os.path.exists(INPUT_FILE):
        print(f"  {RD}❌  {INPUT_FILE} not found{R}"); sys.exit(1)

    env,n,emin,emax,emean = load_and_envelope()
    lib,so                = compile_c()

    if lib:
        bth,bp,bg,bnb = sweep_c(lib,env,n)
        bits = decode_py(list(env),bth,bp,bg)
        try: os.unlink(so)
        except: pass
    else:
        bth,bp,bg,bnb = sweep_py(list(env),n)
        bits = decode_py(list(env),bth,bp,bg)

    if not bits:
        print(f"  {RD}0 bits — signal too weak or params out of range{R}")
        print(f"  {YL}Run python3 diagnose.py for guidance{R}"); sys.exit(1)

    title("OUTPUT",YL)
    bd=bits_to_bytes(bits[:])
    with open(OUTPUT_FILE,'wb') as f: f.write(bd)
    print(f"  {GR}✓{R}  {len(bits)} bits → {len(bd)} bytes → {BOLD}{OUTPUT_FILE}{R}\n")
    hexdump(bd)
    print()
    compare_ref(bd)
    print(f"{GR}{BOLD}  ✅  Done.{R}\n")

if __name__ == "__main__":
    main()
EOF
'''

'''bash
python3 auto_decode.py
'''
