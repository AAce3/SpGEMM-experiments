"""Regression checks: python3 tests/test_combblas_numerics.py EXE [MPI_RANKS]."""
import os, sys, subprocess, tempfile, random, json
from pathlib import Path
exe = str(Path(sys.argv[1]).resolve())
ranks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
launcher = ["mpiexec", "-n", str(ranks)] if ranks > 1 else []
rng = random.Random(7)
n = 32
cases = [
 ('general', 'real general', [(0,0,2),(0,1,3),(1,0,-1),(1,1,4)], True),
 ('duplicates', 'real general', [(0,0,2),(0,0,3),(0,1,4),(1,0,1)], True),
 ('symmetric', 'real symmetric', [(0,0,2),(1,0,3),(1,1,4)], True),
 ('pattern', 'pattern symmetric', [(1,0,1),(2,1,1),(3,3,1)], True),
 ('empty', 'real general', [], True),
 ('random', 'real general', [(i,j,rng.randint(-4,5)) for i in range(n) for j in range(n) if rng.random()<.2], True),
 ('missing_newline', 'real general', [(0,0,12)], False),
 ('skew_symmetric', 'real skew-symmetric', [(1,0,3)], True),
]
results=[]
with tempfile.TemporaryDirectory(prefix='combblas-numerics-') as td:
 for name,kind,entries,newline in cases:
  a=[[0.]*n for _ in range(n)]
  for i,j,v in entries:
   a[i][j]+=v
   if 'symmetric' in kind and i!=j: a[j][i]+= -v if 'skew' in kind else v
  expected=[[sum(a[i][k]*a[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
  inp=Path(td)/f'{name}.mtx'
  lines=[f'%%MatrixMarket matrix coordinate {kind}',f'{n} {n} {len(entries)}']
  lines += [f'{i+1} {j+1}'+('' if 'pattern' in kind else f' {v}') for i,j,v in entries]
  inp.write_text('\n'.join(lines)+ ('\n' if newline else ''))
  for threads in (1,4):
   out=Path(td)/f'{name}-{threads}.mtx'
   env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE')
   run=subprocess.run(launcher + [exe,str(inp),'--output',str(out)],env=env,capture_output=True,text=True,timeout=30)
   result={'case':name,'threads':threads,'exit':run.returncode}
   if run.returncode==0 and out.exists():
    data=[l for l in out.read_text().splitlines() if l and not l.startswith('%')]
    actual=[[0.]*n for _ in range(n)]
    for l in data[1:]:
     i,j,v=l.split();actual[int(i)-1][int(j)-1]+=float(v)
    result['max_abs_error']=max(abs(actual[i][j]-expected[i][j]) for i in range(n) for j in range(n))
   else:result['stderr_tail']=run.stderr
   results.append(result)
print(json.dumps(results,indent=2))

for result in results:
 if result['case'] == 'skew_symmetric':
  assert result['exit'] != 0, result
  assert 'Unsupported Matrix Market format' in result.get('stderr_tail', ''), result
 else:
  assert result['exit'] == 0 and result.get('max_abs_error') == 0, result
print('All numerical and input rejection checks passed.')
