"""Seeded, bounded CPU prefetch with identical consumed batches across workers."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import pickle
import random
import time

import numpy as np
import torch

_LOADER=None
_DECISIONS=None


def _fix_blur_math(transform):
    if transform is None:return
    if type(transform).__name__=="GaussianBlurTransform":transform.benchmark=False
    for value in vars(transform).values():
        if isinstance(value,(list,tuple)):
            for item in value:
                if hasattr(item,"__dict__") and type(item).__module__.startswith("batchgeneratorsv2"):_fix_blur_math(item)
        elif hasattr(value,"__dict__") and type(value).__module__.startswith("batchgeneratorsv2"):
            _fix_blur_math(value)


def _initialize(loader_bytes):
    global _LOADER,_DECISIONS
    torch.set_num_threads(1)
    _LOADER=pickle.loads(loader_bytes)
    # Runtime FFT/non-FFT benchmarking must not choose a different numerical
    # augmentation kernel based on process scheduling or CPU load.
    _fix_blur_math(getattr(_LOADER,"transforms",None))
    decision=_LOADER.get_do_oversample
    _DECISIONS=[]
    def observe(index):
        value=decision(index)
        _DECISIONS.append(bool(value))
        return value
    _LOADER.get_do_oversample=observe


def _produce(token):
    seed,dataset,fold,phase,index=token
    values=np.random.SeedSequence([seed,dataset,fold,phase,index]).generate_state(3)
    random.seed(int(values[0]));np.random.seed(int(values[1]));torch.manual_seed(int(values[2]))
    _DECISIONS.clear()
    batch=next(_LOADER)
    data=batch["data"].numpy() if isinstance(batch["data"],torch.Tensor) else batch["data"]
    target=batch["target"]
    target=[v.numpy() if isinstance(v,torch.Tensor) else v for v in target] if isinstance(target,list) \
        else target.numpy() if isinstance(target,torch.Tensor) else target
    digest=hashlib.sha256()
    if index<32 or index%1000==0:
        digest.update(data.tobytes())
        for value in target if isinstance(target,list) else [target]:digest.update(value.tobytes())
    return dict(data=data,target=target,keys=list(batch["keys"]),batch_index=index,
                forced_foreground=sum(_DECISIONS),samples=len(_DECISIONS),
                fingerprint=digest.hexdigest() if index<32 or index%1000==0 else None)


class OrderedPrefetch:
    def __init__(self,loader,seed,dataset,fold,phase,workers=4,prefetch=8,start_index=0):
        self.loader_bytes=pickle.dumps(loader,protocol=pickle.HIGHEST_PROTOCOL)
        self.token=(int(seed),int(dataset),int(fold),int(phase))
        self.workers,self.prefetch=workers,prefetch
        if start_index < 0:raise ValueError("start_index must be nonnegative")
        self.index,self.next_submit=int(start_index),int(start_index)
        self.pool=None;self.pending={};self.closed=False
        self.fetch_seconds=[]
        if workers:
            self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                                         initializer=_initialize,initargs=(self.loader_bytes,))
        else:_initialize(self.loader_bytes)

    def __iter__(self):return self

    def __next__(self):
        if self.closed:raise StopIteration
        started=time.perf_counter()
        if self.pool:
            while len(self.pending)<self.prefetch:
                i=self.next_submit;self.next_submit+=1
                self.pending[i]=self.pool.submit(_produce,(*self.token,i))
            payload=self.pending.pop(self.index).result()
        else:
            _initialize(self.loader_bytes)
            payload=_produce((*self.token,self.index))
        self.index+=1
        payload["data"]=torch.from_numpy(payload["data"])
        target=payload["target"]
        payload["target"]=[torch.from_numpy(v) for v in target] if isinstance(target,list) else torch.from_numpy(target)
        self.fetch_seconds.append(time.perf_counter()-started)
        return payload

    def close(self):
        if self.closed:return
        self.closed=True
        for future in self.pending.values():future.cancel()
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
        self.pending.clear()

    def __del__(self):
        if not getattr(self,"closed",True):
            try:self.close()
            except Exception:pass
