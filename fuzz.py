import os

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
from functools import reduce
from math import sqrt
def factors(n):
        step = 2 if n%2 else 1
        return set(reduce(list.__add__,
                    ([i, n//i] for i in range(1, int(sqrt(n))+1, step) if n % i == 0)))

compute_capability = torch.cuda.get_device_capability()

MIN_B = 1
MAX_B = 64
MIN_SEQLEN_Q = 1
MIN_SEQLEN_KV = 1
MAX_SEQLEN_Q = 2**16
MAX_SEQLEN_KV = 2**16
MIN_HEAD = 1
MAX_HEAD = 2048
MIN_DQK = 1
MAX_DQK = 256 if compute_capability == (9, 0) or compute_capability == (10, 0) else 128
MIN_DMOD = 8
MIN_DV = 1
MAX_DV = 256 if compute_capability == (9, 0) or compute_capability == (10, 0) else 128
MAX_ELEM = 2**25
CHECK_REF = True
REF_DTYPE = torch.half

#i = 0
#num_gpus = torch.cuda.device_count()
num_gpus = int(os.environ["WORLD_SIZE"])
device = int(os.environ["LOCAL_RANK"])
dtypes = [torch.half, torch.bfloat16]
i = device

while True:
    # device = i % num_gpus
    dtype = dtypes[torch.randint(low=0, high=len(dtypes), size=(1,)).item()]
    b = torch.randint(low=MIN_B, high=MAX_B+1, size=(1,)).item()
    high_sq = int(min(MAX_SEQLEN_Q, MAX_ELEM/b) + 1)
    high_skv = int(min(MAX_SEQLEN_KV, MAX_ELEM/b) + 1)
    if high_sq <= MIN_SEQLEN_Q or high_skv <= MIN_SEQLEN_KV:
        continue
    s_q = torch.randint(low=MIN_SEQLEN_Q, high=high_sq, size=(1,)).item()
    s_kv = torch.randint(low=MIN_SEQLEN_KV, high=high_skv, size=(1,)).item()
    high_hq = int(min(MAX_HEAD, MAX_ELEM/(b*s_q)) + 1)
    h_q = torch.randint(low=MIN_HEAD, high=high_hq, size=(1,)).item()
    h_kv_choices = list(factors(h_q))
    h_k = h_kv_choices[torch.randint(low=0, high=len(h_kv_choices), size=(1,)).item()]
    same_hk_hq = torch.randint(low=0, high=2, size=(1,)).item() == 1
    if same_hk_hq:
        h_v = h_k
    else:
        h_v = h_kv_choices[torch.randint(low=0, high=len(h_kv_choices), size=(1,)).item()]
    high_dqk = int(min(MAX_DQK, MAX_ELEM/(b*s_q*h_q), MAX_ELEM/(b*s_kv*h_k))//MIN_DMOD) + 1
    high_dv = int(min(MAX_DV, MAX_ELEM/(b*s_kv*h_v))//MIN_DMOD) + 1
    if high_dqk <= MIN_DQK//MIN_DMOD + 1 or high_dv <= MIN_DV//MIN_DMOD + 1:
        continue
    d_qk = torch.randint(low=MIN_DQK//MIN_DMOD + 1, high=high_dqk, size=(1,)).item() * MIN_DMOD
    d_v = torch.randint(low=MIN_DV//MIN_DMOD + 1, high=high_dv, size=(1,)).item() * MIN_DMOD
    out_numel = b * s_q * h_q * d_v;
    if out_numel > MAX_ELEM:
        continue
    q_permute = list(torch.randperm(3)) + [3]
    q_reverse = [q_permute.index(i) for i in range(4)]
    k_permute = list(torch.randperm(3)) + [3]
    k_reverse = [k_permute.index(i) for i in range(4)]
    v_permute = list(torch.randperm(3)) + [3]
    v_reverse = [v_permute.index(i) for i in range(4)]
    use_dropout = torch.randint(low=0, high=2, size=(1,)).item() == 1
    dropout_p = 0.0
    if use_dropout:
      dropout_p = torch.rand(1).item()

    grad_permute = list(torch.randperm(3)) + [3]
    grad_reverse = [grad_permute.index(i) for i in range(4)]

    case_str = (f"GPU: {device} case: {i}\n"
        f"dtype: {dtype}\n"
        f"Q {[b, h_q, s_q, d_qk]} numel {b*s_q*h_q*d_qk} layout {q_permute}\n"
        f"K {[b, h_k, s_kv, d_qk]} numel {b*s_kv*h_k*d_qk} layout {k_permute}\n"
        f"V {[b, h_v, s_kv, d_v]} numel {b*s_kv*h_v*d_v} layout {v_permute}\n"
        f"O {[b, h_q, s_q, d_v]} numel {out_numel}\n"
        f"dO {[b, h_q, s_q, d_v]} numel {out_numel} layout {grad_permute}\n"
        f"dropout p: {dropout_p}\n")

    print(f"GPU: {device} case: {i}\n")
   
    qfillshape = [[b, h_q, s_q, d_qk][idx] for idx in q_permute]
    kfillshape = [[b, h_k, s_kv, d_qk][idx] for idx in k_permute]
    vfillshape = [[b, h_v, s_kv, d_v][idx] for idx in v_permute]
    q = torch.randn(qfillshape, dtype=dtype, device=f'cuda:{device}', requires_grad=True).permute(q_reverse).detach()
    q.requires_grad=True
    k = torch.randn(kfillshape, dtype=dtype, device=f'cuda:{device}', requires_grad=True).permute(k_reverse).detach()
    k.requires_grad=True
    v = torch.randn(vfillshape, dtype=dtype, device=f'cuda:{device}', requires_grad=True).permute(v_reverse).detach()
    v.requires_grad=True
    assert q.is_leaf
    assert k.is_leaf
    assert v.is_leaf

    grad_outputfillshape = [[b, h_q, s_q, d_v][idx] for idx in grad_permute]
    grad_output = torch.randn(grad_outputfillshape, dtype=dtype, device=f'cuda:{device}').permute(grad_reverse)

    ref_ok = False
    try:
        if CHECK_REF:
          q_ref = q.detach().to(dtype)
          k_ref = k.detach().to(dtype)
          v_ref = v.detach().to(dtype)
          q_ref.requires_grad=True
          k_ref.requires_grad=True
          v_ref.requires_grad=True
          try:
              with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                  out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, enable_gqa=True)
                  grad_output_ref = grad_output.to(dtype)
                  out_ref.backward(grad_output_ref)
                  ref_ok = True
          except RuntimeError as e:
              print(f"GPU: {device} skipping error in computing ref...")
    except torch.OutOfMemoryError as e:
        print(f"GPU: {device} hit OOM while trying to compute ref...")
        
    try: 
        if CHECK_REF and ref_ok:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION): 
                assert q.is_leaf
                assert k.is_leaf
                assert v.is_leaf
                out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
                torch.testing.assert_close(out, out_ref.to(dtype), atol=7e-3, rtol=7e-3)
                out.backward(grad_output)
                assert k.grad is not None
                assert v.grad is not None
                assert q.grad is not None
                assert q_ref.grad is not None
                grad_atol = 5e-2 if dtype is torch.float16  else 5e-1
                grad_rtol = 5e-3 if dtype is torch.float16 else 5e-1
                torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=grad_atol, rtol=grad_rtol)
                torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=grad_atol, rtol=grad_rtol)
                torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=grad_atol, rtol=grad_rtol)

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, enable_gqa=True).sum().backward()
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p, enable_gqa=True).sum().backward()
    except torch.OutOfMemoryError as e:
        print(f"GPU: {device} hit OOM, assuming it was a cuDNN workspace...")
        continue
    except RuntimeError as e:
        if "No available kernel." in str(e):
            print(f"GPU: {device} hit unsupported heuristic case, assuming it's seqlen 1 droppout...")
            print(case_str)
            continue
        elif "decode only mode" in str(e):
            print(f"GPU: {device} hit decode only error, assuming it's seqlen 1 v9.9...")
            print(case_str)
            continue
        else:
            print("FAILED case:", case_str, str(e))
            raise e
    except AssertionError as e:
        if torch.isnan(out_ref).any() or torch.isnan(q_ref.grad).any() or torch.isnan(k_ref.grad).any() or torch.isnan(v_ref.grad).any():
            print(f"GPU: {device} NaNs in reference, assuming unstable fp16 case, skipping...")
        else:
            print("FAILED case:", case_str, str(e))
            raise e
    except Exception as e:
        print("FAILED case:", case_str, str(e))
        raise e
    i += num_gpus
