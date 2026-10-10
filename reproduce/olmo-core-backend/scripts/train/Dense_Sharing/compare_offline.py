import wandb,json,statistics,math
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util
wandb._assert_is_internal_process=True
import argparse
parser=argparse.ArgumentParser(description='Compare local copies named dense1,dense2,S1,S2,S3.wandb')
parser.add_argument('data_dir',type=Path)
root=parser.parse_args().data_dir

def read(name):
 s=DataStore();s.open_for_scan(str(root/(name+'.wandb')));out={}
 while True:
  b=s.scan_data()
  if b is None:break
  r=wandb_internal_pb2.Record.FromString(b)
  if r.WhichOneof('record_type')!='history':continue
  d=proto_util.dict_from_proto_list(r.history.item)
  if 'train/CE loss' in d:out[int(d.get('_step',r.history.step.num))]=d['train/CE loss']
 return out
x,y=read('dense1'),read('dense2')
runs={'Dense':{i:x[i] if i<=3576 else y[i] for i in range(1,14307)}}
runs.update({k:read(k) for k in ('S1','S2','S3')})
common=set(range(1,14307)).intersection(*(set(v) for v in runs.values()))
bad={s for v in runs.values() for s,x in v.items() if not math.isfinite(x) or x<=0}
clean=sorted(common-bad)
result={'common_count':len(common),'excluded':sorted(common&bad),'runs':{}}
for k,v in runs.items():
 tail=[s for s in clean if s>=13307]
 result['runs'][k]={'rows':len(v),'missing':sorted(set(range(1,14307))-set(v)),'full_mean':statistics.mean(v[s] for s in clean),'tail_mean':statistics.mean(v[s] for s in tail),'tail_n':len(tail),'final':v.get(14306),'raw_full_mean':statistics.mean(v[s] for s in sorted(common))}
print(json.dumps(result,indent=2));(root/'comparison.json').write_text(json.dumps(result,indent=2))
