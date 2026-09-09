"""Model-free state-transition diagnostic with the serving page/stride geometry."""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

if sys.flags.optimize:
    raise RuntimeError('optimized Python is forbidden')
OUT = Path(sys.argv[1]); OUT.mkdir(exist_ok=False)
with (OUT / 'raw.jsonl').open('x') as raw:
    def emit(event, **values):
        raw.write(json.dumps({'time_unix': time.time(), 'event': event, **values}, allow_nan=False) + '\n'); raw.flush()
    def require(value, message):
        if not value: raise ValueError(message)
    emit('start', scope='synthetic state transitions, not model fidelity or context qualification')
    import torch
    torch.set_num_threads(1)
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.models.glm5next.nvidia import kda

    channels, heads, dim, page = 24576, 64, 128, 4456448
    conv_bytes, state_bytes = 147456, 4194304
    ids_list = [1, 3, 4, 6]
    host_pages = torch.full((8, 1, 1, page), 65, dtype=torch.int8, pin_memory=True)
    def bind(pages):
        owner = SimpleNamespace(get_state_shape=lambda: ((3, channels), (heads, dim, dim)),
                                get_state_dtype=lambda: (torch.bfloat16, torch.float32))
        MambaBase.bind_kv_cache(owner, pages)
        return owner.kv_cache
    host_conv, host_state = bind(host_pages)
    for i, slot in enumerate(ids_list):
        host_conv[slot].fill_((i + 1) / 32)
        host_state[slot].copy_(torch.eye(dim)[None].expand(heads, -1, -1) * ((i + 1) / 64))
    initial_pages = host_pages.clone()
    pages = host_pages.to('cuda', non_blocking=True)
    conv_storage, state = bind(pages)
    conv = conv_storage.transpose(-1, -2)
    require(list(conv.stride()) == [page // 2, 1, channels], 'unexpected conv stride')
    require(list(state.stride()) == [page // 4, dim * dim, dim, 1], 'unexpected recurrent stride')
    ids_host = torch.tensor(ids_list, dtype=torch.int32).pin_memory()
    starts_host = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32).pin_memory()
    flags_host = torch.ones(4, dtype=torch.bool).pin_memory()
    weights_host = (((torch.arange(channels)[:, None] + torch.arange(4)[None]) % 5 - 2).float() / 8).pin_memory()
    ids, starts, flags, weights = [x.to('cuda', non_blocking=True) for x in (ids_host, starts_host, flags_host, weights_host)]
    nums, batch_ptr, offsets = compute_causal_conv1d_metadata(starts_host, device=torch.device('cuda'))
    require(nums[8]['mlist'].is_pinned() and nums[8]['offsetlist'].is_pinned(), 'unpinned metadata')
    metadata = SimpleNamespace(nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=offsets)
    torch.cuda.synchronize()
    emit('geometry', conv_stride=list(conv.stride()), state_stride=list(state.stride()),
         page_bytes=page, selected_state_ids=ids_list, query_starts=starts_host.tolist())
    torch.save({'pages': host_pages, 'ids': ids_host, 'starts': starts_host, 'weights': weights_host}, OUT / 'initial.pt')
    checks = []
    for iteration in range(2):
        # Match the wider fused projection view; preserve all 320 trailing columns.
        backing_host = torch.full((128, channels + 320), 3.25, dtype=torch.bfloat16, pin_memory=True)
        for seq in range(4):
            values = ((torch.arange(32)[:, None] + torch.arange(channels)[None] % 16 + seq * 4 + iteration) % 23 - 11).float() / 32
            backing_host[seq * 32:(seq + 1) * 32, :channels].copy_(values)
        backing = backing_host.to('cuda', non_blocking=True)
        before_conv = conv_storage.cpu()
        before_recurrent = state.cpu()
        torch.cuda.synchronize(); emit('before_convolution', iteration=iteration)
        output = causal_conv1d_fn(backing[:, :channels].transpose(0, 1), weights, None,
            conv_states=conv, query_start_loc=starts, cache_indices=ids,
            has_initial_state=flags, activation='silu', metadata=metadata).transpose(0, 1)
        torch.cuda.synchronize(); emit('after_convolution', iteration=iteration)
        require(torch.equal(state.cpu(), before_recurrent), 'convolution changed recurrent state')
        after_conv_history = conv_storage.cpu()
        observed = output.cpu().float()
        reference = torch.empty_like(observed)
        for seq, slot in enumerate(ids_list):
            inputs = backing_host[seq * 32:(seq + 1) * 32, :channels].float()
            joined = torch.cat((before_conv[slot].float(), inputs))
            z = sum(joined[tap:tap + 32] * weights_host[:, tap] for tap in range(4))
            reference[seq * 32:(seq + 1) * 32] = z.double().sigmoid().mul(z.double()).float()
            require(torch.equal(conv_storage[slot].cpu(), backing_host[(seq + 1) * 32 - 3:(seq + 1) * 32, :channels]), 'conv history differs')
        error = (observed - reference).abs()
        require(bool(torch.isfinite(observed).all() and (error <= 0.001 + 0.01 * reference.abs()).all()), 'convolution oracle differs')
        require(torch.equal(backing.cpu(), backing_host), 'input projection changed')
        initial = kda.gather_initial_states(state, ids, flags)
        torch.cuda.synchronize(); emit('after_state_gather', iteration=iteration)
        require(torch.equal(initial.cpu(), before_recurrent[ids_list]), 'gathered state differs')
        q, key, value = [x.reshape(1, 128, heads, dim) for x in output.split(channels // 3, dim=-1)]
        gate = torch.zeros((1, 128, heads, dim), dtype=torch.bfloat16, device='cuda')
        beta_raw = torch.zeros((128, heads), dtype=torch.bfloat16, device='cuda')
        beta = kda._cast_sigmoid(beta_raw).unsqueeze(0)
        a_log = torch.zeros((1, 1, heads, 1), dtype=torch.float32, device='cuda')
        bias = torch.zeros(heads * dim, dtype=torch.float32, device='cuda')
        torch.cuda.synchronize(); emit('before_recurrence', iteration=iteration)
        result, final = kda.chunk_kda_with_fused_gate(q=q, k=key, v=value, raw_g=gate,
            beta=beta, A_log=a_log, g_bias=bias, initial_state=initial,
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=starts,
            safe_gate=True, lower_bound=-5.0)
        torch.cuda.synchronize(); emit('after_recurrence', iteration=iteration)
        require(bool(torch.isfinite(result).all() and torch.isfinite(final).all()), 'nonfinite recurrence')
        require(torch.equal(state.cpu(), before_recurrent), 'gather or recurrence changed source state')
        kda.scatter_states(state, final, ids)
        torch.cuda.synchronize(); emit('after_state_scatter', iteration=iteration)
        after = pages.cpu()
        require(torch.equal(state.cpu()[ids_list], final.cpu()), 'scattered state differs')
        require(torch.equal(conv_storage.cpu(), after_conv_history), 'recurrent path changed convolution history')
        for slot in set(range(8)) - set(ids_list):
            require(torch.equal(after[slot], initial_pages[slot]), 'unused page changed')
        require(bool((after[:, :, :, conv_bytes + state_bytes:] == 65).all()), 'page padding changed')
        torch.save({'input': backing_host, 'conv': output.cpu(), 'result': result.cpu(), 'final': final.cpu(), 'pages': after}, OUT / f'iteration-{iteration}.pt')
        row = {'iteration': iteration, 'convolution_max_absolute_error': float(error.max()),
               'state_transfer_exact': True, 'unused_pages_and_padding_unchanged': True,
               'cross_region_preservation': True,
               'recurrence_finite': True}
        checks.append(row); emit('checked', **row)
    summary = {'scope': 'two synthetic state transitions, no model weights',
               'diagnostic_checks': 'PASS', 'crash_attribution': 'NO_RESULT', 'checks': checks,
               'limits': 'KDA full numerical fidelity and the originating model failure are not established; JIT preparation, if any, is not a frozen confirmation.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
