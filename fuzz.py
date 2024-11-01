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

MIN_B = 1
MAX_B = 128
MIN_SEQLEN_Q = 2
MIN_SEQLEN_KV = 2
MAX_SEQLEN_Q = 2**17
MAX_SEQLEN_KV = 2**17
MIN_HEAD = 1
MAX_HEAD = 2048
MIN_DQK = 1
MAX_DQK = 128
MIN_DMOD = 8
MIN_DV = 1
MAX_DV = 128
MAX_ELEM = 2**29
CHECK_FP32REF = True

i = 0
num_gpus = torch.cuda.device_count()

while True:
    device = i % num_gpus
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
    h_v = h_kv_choices[torch.randint(low=0, high=len(h_kv_choices), size=(1,)).item()]
    high_dqk = int(min(MAX_DQK, MAX_ELEM/(b*s_q*h_q), MAX_ELEM/(b*s_kv*h_k))//MIN_DMOD) + 1
    high_dv = int(min(MAX_DV, MAX_ELEM/(b*s_kv*h_v))//MIN_DMOD) + 1
    if high_dqk <= MIN_DQK or high_dv <= MIN_DV:
        continue
    d_qk = torch.randint(low=MIN_DQK//MIN_DMOD + 1, high=high_dqk, size=(1,)).item() * MIN_DMOD
    d_v = torch.randint(low=MIN_DV//MIN_DMOD + 1, high=high_dv, size=(1,)).item() * MIN_DMOD
    out_numel = b * s_q * h_q * d_v;
    if out_numel > MAX_ELEM:
        continue
    i += 1
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


    print(f"GPU: {device} case: {i}\n"
        f"Q {[b, h_q, s_q, d_qk]} numel {b*s_q*h_q*d_qk} layout {q_permute}\n"
        f"K {[b, h_k, s_kv, d_qk]} numel {b*s_kv*h_k*d_qk} layout {k_permute}\n"
        f"V {[b, h_v, s_kv, d_v]} numel {b*s_kv*h_v*d_v} layout {v_permute}\n"
        f"O {[b, h_q, s_q, d_v]} numel {out_numel}\n"
        f"dO {[b, h_q, s_q, d_v]} numel {out_numel} layout {grad_permute}\n"
        f"dropout p: {dropout_p}\r")
   
    qfillshape = [[b, h_q, s_q, d_qk][idx] for idx in q_permute]
    kfillshape = [[b, h_k, s_kv, d_qk][idx] for idx in k_permute]
    vfillshape = [[b, h_v, s_kv, d_v][idx] for idx in v_permute]
    q = torch.randn(qfillshape, dtype=torch.half, device=f'cuda:{device}', requires_grad=True).permute(q_reverse).detach()
    q.requires_grad=True
    k = torch.randn(kfillshape, dtype=torch.half, device=f'cuda:{device}', requires_grad=True).permute(k_reverse).detach()
    k.requires_grad=True
    v = torch.randn(vfillshape, dtype=torch.half, device=f'cuda:{device}', requires_grad=True).permute(v_reverse).detach()
    v.requires_grad=True
    assert q.is_leaf
    assert k.is_leaf
    assert v.is_leaf

    grad_outputfillshape = [[b, h_q, s_q, d_v][idx] for idx in grad_permute]
    grad_output = torch.randn(grad_outputfillshape, dtype=torch.half, device=f'cuda:{device}').permute(grad_reverse)

    ref_ok = False
    try:
        if CHECK_FP32REF:
          q_ref = q.detach().float()
          k_ref = k.detach().float()
          v_ref = v.detach().float()
          q_ref.requires_grad=True
          k_ref.requires_grad=True
          v_ref.requires_grad=True
          print(q_ref.shape, k_ref.shape, v_ref.shape) 
          out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, enable_gqa=True)
          grad_output_ref = grad_output.float()
          out_ref.backward(grad_output_ref)
          ref_ok = True
    except torch.OutOfMemoryError as e:
        print("hit OOM while trying to compute ref...")
        
    try: 
        if CHECK_FP32REF and ref_ok:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION): 
                assert q.is_leaf
                assert k.is_leaf
                assert v.is_leaf
                out = F.scaled_dot_product_attention(q, k, v)
                torch.testing.assert_close(out, out_ref.half(), atol=1e-3, rtol=1e-3)
                out.backward(grad_output)
                assert k.grad is not None
                assert v.grad is not None
                assert q.grad is not None
                assert q_ref.grad is not None

                torch.testing.assert_close(q.grad, q_ref.grad.half(), atol=1e-3, rtol=1e-3)
                torch.testing.assert_close(k.grad, k_ref.grad.half(), atol=1e-3, rtol=1e-3)
                torch.testing.assert_close(v.grad, v_ref.grad.half(), atol=1e-3, rtol=1e-3)

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p).sum().backward()
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p).sum().backward()
    except torch.OutOfMemoryError as e:
        print("hit OOM, assuming it was a cuDNN workspace...")
        continue

