"""Bounded native-MLX Qwen3-1.7B experiment; official weights, frozen synthetic data."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs_mlx'
BASE = RUN / 'mlx-qwen3-1.7b-4bit'
MAX_RUNTIME_HOURS = 3.0
BUDGET_END = time.monotonic() + MAX_RUNTIME_HOURS * 3600
MAX_LENGTH = 512
LORA = {'rank':16, 'scale':2.0, 'dropout':0.05,
        'keys':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                'mlp.gate_proj','mlp.up_proj','mlp.down_proj']}
sys.path.insert(0, str(ROOT / 'evaluation'))
from evaluate import evaluate, parse_output, paired_family_bootstrap, align


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_budget():
    if time.monotonic() >= BUDGET_END:
        raise TimeoutError('Local execution duration limit reached.')


def data():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    rows={}
    for split in ('train','val','test'):
        path=ROOT/'data'/f'{split}.json'
        if sha(path)!=manifest['sha256'][path.name]:raise ValueError('Frozen data changed: '+split)
        rows[split]=json.loads(path.read_text())
    return rows


def messages(row):
    return [{'role':'system','content':row['system']},
            {'role':'user','content':row['instruction']+'\n'+row['input']}]


def encode(tokenizer,row):
    prompt=tokenizer.apply_chat_template(messages(row),tokenize=True,return_dict=False,
                                         add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(messages(row)+[{'role':'assistant','content':row['output']}],
                                       tokenize=True,return_dict=False,enable_thinking=False)
    if full[:len(prompt)]!=prompt:raise ValueError('Prompt/target prefix mismatch: '+row['id'])
    if len(full)>MAX_LENGTH:raise ValueError('Truncation prohibited: '+row['id'])
    if not tokenizer.decode(full[len(prompt):]).startswith(row['output']):
        raise ValueError('Assistant target mismatch: '+row['id'])
    return full,len(prompt)


def assistant_mask(lengths, width):
    import mlx.core as mx
    positions=mx.arange(1,width+1)
    # Targets are tokens[1:]; positions are their original sequence positions.
    # The strict upper boundary excludes the first padded token at original len.
    return (positions>=lengths[:,0:1]) & (positions<lengths[:,1:2])


def assistant_loss(model,batch,lengths):
    import mlx.core as mx
    import mlx.nn as nn
    logits=model(batch[:,:-1]); targets=batch[:,1:]
    mask=assistant_mask(lengths,targets.shape[1]); count=mask.sum()
    losses=nn.losses.cross_entropy(logits,targets).astype(mx.float32)
    return (losses*mask).sum()/count,count


def preflight():
    from transformers import AutoTokenizer
    import mlx.core as mx
    from mlx_lm.tuner.trainer import iterate_batches
    t=AutoTokenizer.from_pretrained(RUN/'official-base',local_files_only=True,trust_remote_code=False)
    rows=data(); stats={}
    for split,records in rows.items():
        encoded=[encode(t,r) for r in records]
        # Every source record, including both masks and padding, is checked.
        for tokens,offset in encoded:
            batch,lengths=next(iterate_batches([(tokens,offset)],1,MAX_LENGTH))
            expected=list(range(offset,len(tokens)))
            mask=assistant_mask(lengths,batch.shape[1]-1).tolist()[0]
            observed=[i+1 for i,active in enumerate(mask) if active]
            if expected!=observed:raise ValueError('Assistant mask or padding boundary mismatch')
            if batch.tolist()[0][:len(tokens)]!=tokens:raise ValueError('Truncated token IDs')
        stats[split]={'records':len(encoded),'min_tokens':min(len(x[0]) for x in encoded),
                      'max_tokens':max(len(x[0]) for x in encoded),'all_assistant_masks_verified':True,
                      'all_tokens_preserved':True,'targets_include_terminal_chat_eos_and_newline':True}
    report={'passed':True,'data_sha256':json.loads((ROOT/'data/manifest.json').read_text())['sha256'],
            'tokenizer_revision':json.loads((RUN/'base_identity.json').read_text())['revision'],
            'enable_thinking':False,'max_length':MAX_LENGTH,'splits':stats,
            'mlx_lm':importlib.metadata.version('mlx-lm'),'mlx':importlib.metadata.version('mlx')}
    write(RUN/'preflight.json',report); print('PREFLIGHT',json.dumps(report),flush=True)


def convert():
    from mlx_lm.convert import convert as mlx_convert
    if not BASE.exists():
        mlx_convert(str(RUN/'official-base'),str(BASE),quantize=True,q_bits=4,q_group_size=64,
                    q_mode='affine',dtype='bfloat16',trust_remote_code=False)
    config=json.loads((BASE/'config.json').read_text())
    write(RUN/'conversion.json',{'base_identity':json.loads((RUN/'base_identity.json').read_text()),
                               'quantization':config.get('quantization'),
                               'weights':{p.name:sha(p) for p in BASE.glob('*.safetensors')}})
    print('CONVERSION_COMPLETE',flush=True)


def inference(kind):
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    rows=data()['test']; target=RUN/f'qwen3-1.7b_{kind}.jsonl'
    if target.exists():raise ValueError('Refusing to overwrite existing predictions: '+str(target))
    kwargs={'adapter_path':str(RUN/'adapter')} if kind=='adapter' else {}
    model,tokenizer=load(str(BASE),tokenizer_config={'trust_remote_code':False},**kwargs)
    model.eval(); predictions=[]; started=time.monotonic()
    with target.open('x') as handle:
        for index,row in enumerate(rows):
            check_budget()
            prompt=tokenizer.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True,
                                                 enable_thinking=False)
            tick=time.perf_counter()
            raw=generate(model,tokenizer,prompt=prompt,max_tokens=96,sampler=make_sampler(temp=0),verbose=False).strip()
            elapsed=time.perf_counter()-tick
            item={'id':row['id'],'raw_output':raw,'prediction':parse_output(raw)[0],
                  'latency_seconds':elapsed,'generated_tokens':len(tokenizer.encode(raw,add_special_tokens=False))}
            predictions.append(item);handle.write(json.dumps(item,ensure_ascii=False)+'\n');handle.flush()
            if (index+1)%25==0:print(f'{kind.upper()} {index+1}/300 elapsed={time.monotonic()-started:.1f}s',flush=True)
    metrics=evaluate(rows,predictions);write(RUN/f'qwen3-1.7b_{kind}_metrics.json',metrics)
    write(RUN/f'{kind}_receipt.json',{'completed':True,'records':len(predictions),'seconds':time.monotonic()-started,
                                    'peak_memory_bytes':mx.get_peak_memory(),'prediction_sha256':sha(target),
                                    'backend':'mlx_lm','temperature':0,'max_new_tokens':96,'enable_thinking':False})
    print(kind.upper()+'_COMPLETE',json.dumps({k:metrics[k] for k in ['joint_accuracy','macro_f1','strict_json_validity']}),flush=True)
    del model,tokenizer;mx.clear_cache()


def train_adapter():
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import train,TrainingArgs,evaluate as evaluate_loss
    from mlx_lm.tuner.callbacks import TrainingCallback
    if not json.loads((RUN/'base_receipt.json').read_text())['completed']:
        raise ValueError('Complete baseline required before training')
    adapter=RUN/'adapter'
    if adapter.exists():raise ValueError('Existing adapter; refusing a second training run')
    adapter.mkdir()
    model,t=load(str(BASE),tokenizer_config={'trust_remote_code':False})
    rows=data();train_rows=[encode(t,r) for r in rows['train']];valid_rows=[encode(t,r) for r in rows['val']]
    mx.random.seed(42);np.random.seed(42);model.freeze();linear_to_lora_layers(model,-1,LORA)
    parameters=dict(tree_flatten(model.trainable_parameters()))
    trainable=sum(x.size for x in parameters.values())
    config={'fine_tune_type':'lora','num_layers':-1,'lora_parameters':LORA,
            'model':str(BASE),'base_model_name_or_path':'Qwen/Qwen3-1.7B','mask_prompt':True,
            'batch_size':1,'grad_accumulation_steps':16,'iters':2000,'epochs':2,
            'learning_rate':1e-4,'max_seq_length':512,'seed':42,
            'optimizer':'mlx_adamw','weight_decay':0.0,'max_grad_norm':1.0,
            'warmup_updates':10,'schedule':'linear_decay_after_warmup','updates':125}
    write(adapter/'adapter_config.json',config);write(RUN/'training_config.json',config)
    class ClippedAdamW(optim.AdamW):
        def apply_gradients(self,gradients,parameters):
            gradients,_=optim.clip_grad_norm(gradients,1.0)
            return super().apply_gradients(gradients,parameters)
    schedule=optim.join_schedules([optim.linear_schedule(0,1e-4,10),optim.linear_schedule(1e-4,0,115)],[10])
    optimizer=ClippedAdamW(learning_rate=schedule,weight_decay=0.0)
    class Callback(TrainingCallback):
        best=float('inf')
        best_iteration=None
        def event(self,kind,info):
            check_budget()
            with (RUN/'training_events.jsonl').open('a') as handle:
                handle.write(json.dumps({'kind':kind,**info})+'\n')
            print('TRAIN_EVENT',kind,json.dumps(info),flush=True)
        def on_train_loss_report(self,info):self.event('train',info)
        def on_val_loss_report(self,info):
            self.event('validation',info)
            if info['val_loss']<self.best:
                self.best=info['val_loss'];self.best_iteration=info['iteration']
                mx.save_safetensors(str(adapter/'best.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    callback=Callback();started=time.monotonic()
    args=TrainingArgs(batch_size=1,iters=2000,val_batches=-1,steps_per_report=100,
                      steps_per_eval=400,steps_per_save=400,max_seq_length=512,
                      adapter_file=str(adapter/'adapters.safetensors'),grad_checkpoint=True,
                      grad_accumulation_steps=16)
    train(model,optimizer,train_rows,valid_rows,args=args,loss=assistant_loss,training_callback=callback)
    final_loss=evaluate_loss(model,valid_rows,batch_size=1,num_batches=-1,max_seq_length=512,loss=assistant_loss)
    callback.on_val_loss_report({'iteration':2000,'val_loss':final_loss,'final_after_update':True})
    (adapter/'adapters.safetensors').replace(adapter/'final.safetensors')
    (adapter/'adapters.safetensors').write_bytes((adapter/'best.safetensors').read_bytes())
    receipt={'completed':True,'micro_steps':2000,'optimizer_updates':125,'epochs':2,'train_examples':1000,
             'trainable_parameters':trainable,'best_validation_loss':callback.best,
             'best_micro_step':callback.best_iteration,'selection':'minimum full validation loss; no test selection',
             'seconds':time.monotonic()-started,'peak_memory_bytes':mx.get_peak_memory(),
             'adapter_sha256':sha(adapter/'adapters.safetensors'),'format':'native mlx_lm adapter; not PEFT'}
    write(RUN/'training_receipt.json',receipt);print('TRAINING_COMPLETE',json.dumps(receipt),flush=True)
    del model,optimizer;mx.clear_cache()


def package():
    import shutil
    destination=RUN/'release-qwen3-1.7b-mlx'
    destination.mkdir(exist_ok=False)
    for name in ('adapters.safetensors','adapter_config.json'):
        shutil.copy2(RUN/'adapter'/name,destination/name)
    for name in ('base_identity.json','conversion.json','preflight.json','training_config.json','training_receipt.json',
                 'base_receipt.json','adapter_receipt.json','qwen3-1.7b_base_metrics.json','qwen3-1.7b_adapter_metrics.json'):
        shutil.copy2(RUN/name,destination/name)
    shutil.copy2(ROOT/'training_mlx/requirements-lock.txt',destination/'requirements-lock.txt')
    gold=data()['test']
    a=[json.loads(x) for x in (RUN/'qwen3-1.7b_base.jsonl').read_text().splitlines()]
    b=[json.loads(x) for x in (RUN/'qwen3-1.7b_adapter.jsonl').read_text().splitlines()]
    write(destination/'comparison.json',paired_family_bootstrap(align(gold,a),align(gold,b),iterations=2000))
    write(destination/'file_hashes.json',{p.name:sha(p) for p in destination.iterdir() if p.is_file()})
    print('RELEASE_DIRECTORY',destination,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','convert','base','train','adapter','package','execute','all'],required=True)
    parser.add_argument('--max-runtime-hours',type=float,default=MAX_RUNTIME_HOURS,
                        help='Elapsed duration cap for this invocation, at most three hours (default: 3).')
    args=parser.parse_args()
    if not math.isfinite(args.max_runtime_hours) or not 0 < args.max_runtime_hours <= MAX_RUNTIME_HOURS:
        parser.error('--max-runtime-hours must be greater than zero and at most three')
    global BUDGET_END
    started=time.monotonic();seconds=math.ceil(args.max_runtime_hours*3600)
    BUDGET_END=started+args.max_runtime_hours*3600
    def deadline_handler(signum,frame):raise TimeoutError('Local execution duration limit reached')
    signal.signal(signal.SIGALRM,deadline_handler);signal.alarm(seconds)
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    stages={'preflight':preflight,'convert':convert,'base':lambda:inference('base'),
            'train':train_adapter,'adapter':lambda:inference('adapter'),'package':package}
    try:
        for name,fn in stages.items():
            if args.stage in (name,'all') or (args.stage=='execute' and name!='preflight'):
                check_budget();print('STAGE_START',name,flush=True);fn()
    except BaseException as exc:
        write(RUN/'failure.json',{'stage':name,'exception':type(exc).__name__,'detail':str(exc),
                                 'elapsed_seconds':time.monotonic()-started,'complete':False})
        raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
