# Native dummy resolution bound

Media011's immutable trace again ends in API-side dummy video preprocessing
at torch.cat, stopped by the18GiB floor. The16-frame native cap is present in
its frozen launch arguments. Maximum dummy dimensions remain unconstrained.

The next bounded alternative sets only native dummy dimensions to512x512 for
images and video. Real request counts, source media resolution, actual video
frame policy, model, cache, four slots and containment remain unchanged. This
uses existing vLLM dummy options, whose scope is profiling/warm-up; no runtime
patch or new token-path diagnostic is introduced. Engine MM profiling was
already explicitly skipped. This does not qualify large-resolution requests.

The acceptance remains media-bringup-acceptance-001.md, including actual
single/four-image and16-frame video answers, authentication/text/tools and
external memory above18GiB with cgroup swap0. Restore text mode if this fails.
The CPU regression failed before implementation; raw assertion is preserved.
