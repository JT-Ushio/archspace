"""Read-only job/log checks and arch scheduling predictions; run locally."""
import subprocess, shlex, re, json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
HOST='cpu-develop.jitao+root.ailab-llmagent.ws@h.pjlab.org.cn'
JOBS={'S1':'mose-dense-sharing-s1-r2-p1-1x8-pool-1005-13108302','S2':'mose-dense-sharing-s2-r2-p1-1x8-pool-1005-26282060','S3':'mose-dense-sharing-s3-r2-p1-1x8-pool-1005-42104108'}
OUT=Path('/tmp/mose_monitor');OUT.mkdir(exist_ok=True)
def remote(cmd):
 try:
  p=subprocess.run(['ssh','-CAXY',HOST,'zsh -lic '+shlex.quote('conda activate MoSE && '+cmd)],capture_output=True,text=True,timeout=120)
  return p.stdout+p.stderr
 except subprocess.TimeoutExpired:return 'SSH observation timed out; state unknown'
def check(item):
 label,job=item
 raw=remote(f'rjob get --namespace=ailab-llmarchitecture {job} && rjob logs job --namespace=ailab-llmarchitecture {job}')
 (OUT/f'{label}.log').write_text(raw)
 status=re.search(r'\(showname=.*?\): (\w+)',raw);counts=re.search(r'\d+ replicas (\{[^\n]+\})',raw)
 reports=[];current=None
 for line in raw.splitlines():
  m=re.search(r'\[step=(\d+)/14306,epoch=\d+,eta=([^\]]+)',line)
  if m:current={'step':int(m[1]),'eta':m[2]};reports.append(current)
  if current:
   for key,pattern in [('ce',r'train/CE loss=([\d.eE+-]+)'),('grad',r'optim/total grad norm=([\d.eE+-]+)')]:
    m=re.search(pattern,line)
    if m:current[key]=float(m[1])
 return {'label':label,'job':job,'status':status[1] if status else 'UNKNOWN','counts':counts[1] if counts else None,'latest':max((r for r in reports if 'ce' in r),key=lambda x:x['step'],default=None),'events':[l for l in raw.splitlines() if any(w in l for w in ('CUDA out of memory','SIGTERM','Training complete','Traceback (most recent'))][-3:]}
def predict(nodes):
 args=['rjob','submit','--name',f'mose-arch-probe-{nodes}x8','--gpu=8','--memory=262144','--cpu=32','--charged-group=llmarchitecture_gpu','--namespace=ailab-llmarchitecture','--private-machine=group','--task-type=normal','--priority=1','--custom-resources','brainpp.cn/fuse=1','--image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323','-P',str(nodes),'-e','DISTRIBUTED_JOB=true','--host-network=true','--custom-resources','rdma/mlnx_shared=8','--custom-resources','mellanox.com/mlnx_rdma=1']
 for root in ('kuangjiahao-p','wangjie3','jitao','songdemin','ailab-agent-combined','shared-storage-ailab-llmfudan','ailab-hs','auto-eval-pipeline','opencompass-shared'):
  args.append(f'--mount=gpfs://gpfs1/{root}:/mnt/shared-storage-user/'+('ailab-llmagent' if root=='songdemin' else root))
 for root in ('shared-storage-ailab-llmfudan-02','intern-pretrain-shared02','gpfs2-shared-public','trustcyberdata'):
  args.append(f'--mount=gpfs://gpfs2/{root}:/mnt/shared-storage-gpfs2/{root}')
 args+=['--predict-only=true','--','/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/Dense_Sharing/run_s1.sh']
 raw=remote(shlex.join(args));(OUT/f'arch{nodes}.log').write_text(raw)
 return {'nodes':nodes,'summary':[l for l in raw.splitlines() if any(w in l for w in ('核心状态','总副本数量','可调度数量'))]}
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=5) as pool:
  jobs=list(pool.map(check,JOBS.items()));capacity=list(pool.map(predict,[4,2]))
 stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y-%m-%d %H:%M HKT')
 result={'time':stamp,'jobs':jobs,'arch':capacity}
 print(json.dumps(result,ensure_ascii=False,indent=2));(OUT/'status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with Path(__file__).with_name('RUN_STATUS.md').open('a') as f:
  f.write('\n## '+stamp+' — monitoring\n\n')
  for row in jobs:f.write(f"- {row['label']}: {row['status']}; {row['counts']}; {row['latest']}.\n")
  for row in capacity:f.write(f"- Arch {row['nodes']}x8 prediction: {row['summary']}.\n")
