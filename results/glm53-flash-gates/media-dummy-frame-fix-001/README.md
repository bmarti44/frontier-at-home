# Bound startup video allocation to the existing real 16-frame limit

Previous goal turn: PROGRESS. It preserved media009's actual floor failure,
restored text010 with the original key, and reproduced text/tools/four clients.

Observed failure: media009 completed engine warm-up, then API dummy video
preprocessing grew RSS until the 18 GiB watchdog floor killed the unit. Its
unaltered trace and samples are in `../server-bringup-009/`.

Fixed acceptance remains `../media-bringup-acceptance-001.md`. The next bounded
candidate changes only native video dummy options to count=1, num_frames=16.
It preserves real image/video policy, resolution, model/cache bytes, four-slot
geometry, kernel selection, allocator, authentication and containment. The
persistent gap reviewer verified the native path before this change. Dummy
options clamp frame allocation before numpy allocates the video; they do not
replace real input frame enforcement or prove image/video memory safety.

The committed CPU regression first failed on unchanged launch arguments:
`{'image': 4, 'video': 1}`. It also executes the native allocation helper with
an intercepted allocator to check requested dimensions without allocating
pixels. That check is source/configuration evidence only. Actual bounded
media requests and externally observed memory are still required.
