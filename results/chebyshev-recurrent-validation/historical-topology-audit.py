"""Route small circuits on a historical public topology; project, never claim QPU time."""
from pathlib import Path
import argparse,json,time,hashlib,io
import numpy as np
from qiskit import qpy,transpile,ClassicalRegister
from qiskit.transpiler import CouplingMap
from qcenn_mi.recurrent_cost import coherence_readout_rotations
ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--properties',type=Path,required=True);ap.add_argument('--configuration',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
data=json.loads((a.study/'result.json').read_text());props=json.loads(a.properties.read_text());conf=json.loads(a.configuration.read_text());durations={}
units={'s':1.,'ms':1e-3,'us':1e-6,'ns':1e-9}
for gate in props['gates']:
 item=next((p for p in gate['parameters'] if p['name']=='gate_length'),None)
 if item is not None:durations[(gate['gate'],tuple(gate['qubits']))]=item['value']*units[item['unit']]
readout={i:next(p for p in ps if p['name']=='readout_length') for i,ps in enumerate(props['qubits'])}
def latency(circuit,reset_seconds):
 ready=np.zeros(circuit.num_qubits)
 for instruction in circuit.data:
  name=instruction.operation.name;qubits=tuple(circuit.find_bit(q).index for q in instruction.qubits)
  if name=='barrier':continue
  if name=='reset':dt=reset_seconds
  elif name=='measure':p=readout[qubits[0]];dt=p['value']*units[p['unit']]
  else:dt=durations[(name,qubits)]
  ready[list(qubits)]=max(ready[list(qubits)])+dt
 return float(max(ready))
rows=[]
for row in data['cost_cases']:
 if row['status']!='compiled':continue
 file=a.study/f"circuits/converged-k{row['condition']}-{row['method']}.qpy"
 original=qpy.load(io.BytesIO(file.read_bytes()))[0]
 settings=[];compile_time=0
 for i,rotation in enumerate(coherence_readout_rotations(2)):
  circuit=original.copy();circuit.compose(rotation,qubits=[0,1],inplace=True);circuit.add_register(ClassicalRegister(2));circuit.measure([0,1],[0,1])
  start=time.perf_counter();native=transpile(circuit,basis_gates=conf['basis_gates'],coupling_map=CouplingMap(conf['coupling_map']),initial_layout=list(range(circuit.num_qubits)),optimization_level=1,seed_transpiler=42);elapsed=time.perf_counter()-start;compile_time+=elapsed
  filename=f"k{row['condition']}-{row['method']}-setting{i}.qpy"
  buffer=io.BytesIO();qpy.dump(native,buffer);(a.output/filename).write_bytes(buffer.getvalue())
  settings.append({'setting':i,'native_cx':native.count_ops().get('cx',0),'depth':native.depth(),'width':native.num_qubits,'reset_count':native.count_ops().get('reset',0),'qpy_file':filename,'models':[{'reset_microseconds':us,'modeled_seconds':latency(native,us*1e-6)} for us in [.1,1,10]]})
 budget=row['readout_budgets']['0.01'];models=[]
 for us in [.1,1,10]:
  quantum=sum(next(m['modeled_seconds'] for m in s['models'] if m['reset_microseconds']==us) for s in settings)
  for reuse in [1,1000]:
   common=row['common_preprocessing_seconds']+row['phase_synthesis_seconds']+row['transform_seconds']+row['compile_seconds']+row['readout_basis_compile_seconds']
   total=None if budget['total_executions'] is None else (common+compile_time)/reuse+budget['shots_per_setting']*quantum
   models.append({'reset_microseconds':us,'reuse':reuse,'projected_seconds':total})
 rows.append({'condition':row['condition'],'method':row['method'],'settings':settings,'route_compile_seconds':compile_time,'models':models,'input_qpy_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'ideal_shot_budget':budget,'physical_accuracy_certified':False})
payload={'source':data['provenance'],'study_result_sha256':hashlib.sha256((a.study/'result.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'upstream_repository':'Qiskit/qiskit-ibm-runtime','upstream_revision':'52fa07770f5e955000de84a6220e0c8026540428','snapshot_date':props['last_update_date'],'properties_sha256':hashlib.sha256(a.properties.read_bytes()).hexdigest(),'configuration_sha256':hashlib.sha256(a.configuration.read_bytes()).hexdigest(),'physical_hardware_tested':False,'reset_duration_is_assumed':True,'accuracy_includes_calibration_noise':False,'interpretation':'Historical topology/duration projection of an ideal-accuracy workload; no achievable hardware time claimed','cases':rows}
(a.output/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');print('Completed historical routing audit:',len(rows),'cases')
