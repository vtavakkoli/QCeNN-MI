"""Matched primitive-call budget audit using the pinned channel implementation."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
from qcenn_mi.polynomial_recurrent import accelerate_density_channel
from qcenn_mi.polynomial_renewal import renewal_observables
from qcenn_mi.recurrent_density import build_density_channel
from qcenn_mi.validation import provenance

parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
rows=[]
for n in [10,100,1000]:
 for condition in [10,100]:
  name=f'static-n{n}-k{condition}-controlled_nonsymmetric.npz'
  path=args.data/'polynomial-recurrent/inputs'/name
  with np.load(path) as a:matrix,rhs,reference=a['matrix'],a['rhs'],a['reference_solution']
  start=time.perf_counter();base=build_density_channel(matrix,rhs);prep=time.perf_counter()-start
  for m in [0,4,16]:
   primitive_budget=16384;K=primitive_budget//(m+1)
   times=np.unique(np.r_[0,np.arange(1,9),np.rint(np.geomspace(9,K,40))]).astype(int)
   start=time.perf_counter();channel=accelerate_density_channel(base,m)
   trace=renewal_observables(channel,K,times,'reference',reference)
   rows.append({'size':n,'condition':condition,'polynomial_degree':m,'macro_updates':K,
    'primitive_calls':K*(m+1),'primitive_call_budget':primitive_budget,
    'input_file':'polynomial-recurrent/inputs/'+name,
    'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'input_preprocessing_seconds':prep,'response_and_simulation_seconds':time.perf_counter()-start,
    'trajectory':trace,'final':trace[-1],
    'accuracy_passed':trace[-1]['relative_residual']<=1e-6 and 1-trace[-1]['conditional_solution_fidelity']<=1e-6})
result={'source':provenance(Path.cwd()),'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'cost_model':'m+1 residual-unitary calls per macro; extra controls and swaps not discounted',
 'physical_hardware_tested':False,'cases':rows}
dest=args.output/'result.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');os.replace(tmp,dest)
print(json.dumps({'cases':len(rows),'accuracy_passed':sum(r['accuracy_passed'] for r in rows)}))
