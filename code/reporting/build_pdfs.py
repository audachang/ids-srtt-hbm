#!/usr/bin/env python3
"""Build APA PDFs using system tools or the isolated /tmp bibliography runtime."""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--manuscript-dir',type=Path,default=Path(__file__).resolve().parents[2]/'manuscript')
p.add_argument('--runtime',type=Path,default=Path('/tmp/srtt_tex_runtime/root'))
p.add_argument('--force',action='store_true');p.add_argument('--supplement-only',action='store_true')
a=p.parse_args();env=os.environ.copy();r=a.runtime
if (r/'usr/share/texlive/texmf-dist/tex/latex/biblatex/biblatex.sty').is_file():
    env['PATH']=str(r/'usr/bin')+':'+env['PATH']
    env['PERL5LIB']=str(r/'usr/share/perl5')+':'+str(r/'usr/lib/x86_64-linux-gnu/perl5/5.38')
    env['LD_LIBRARY_PATH']=':'.join(sorted({str(x.parent) for x in r.rglob('*.so*')}))
    env['TEXINPUTS']=str(r/'usr/share/texlive/texmf-dist')+'//:'
for name in ['supplement']+([] if a.supplement_only else ['manuscript']):
    cmd=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error']
    if a.force:cmd+=['-g']
    log=a.manuscript_dir/(name+'_build.txt')
    with log.open('w') as f:
        done=subprocess.run(cmd+[name+'.tex'],cwd=a.manuscript_dir,env=env,stdout=f,stderr=subprocess.STDOUT)
    if done.returncode:
        print('\n'.join(log.read_text().splitlines()[-40:]));raise SystemExit(done.returncode)
    bibliography=a.manuscript_dir/(name+'.bbl')
    if bibliography.is_file():
        bibliography.write_text(bibliography.read_text().rstrip()+'\n')
    print('Built',a.manuscript_dir/(name+'.pdf'))
