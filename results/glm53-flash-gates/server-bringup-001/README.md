# First full-model startup: FAIL

The selected weights loaded, but cache allocation failed with NVIDIA
NV_ERR_NO_MEMORY and the external 18 GiB floor stopped the process group.
No serving response was obtained. The host recovered and no GPU process or
port 8015 listener remained. Raw launcher and watchdog records are preserved;
credentials and reproducible warm caches are excluded.

Next bounded alternative: select vLLM's existing `--skip-mm-profiling` switch.
Keep four 262,144-token slots, full model/media support and all memory limits.
Acceptance: healthy authenticated server; unauthenticated models request gets
401; authenticated model listing includes glm-5.3-flash; streamed 2+2 response
completes and contains 4/four. This is a bring-up check, not context, fidelity,
or performance qualification. A memory kill or CUDA allocation failure fails.
