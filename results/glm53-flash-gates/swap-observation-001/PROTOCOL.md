# External swap observation controls

Attempts004/005 retain a failed no-swap-I/O check: one host swap-in page arrived
between the broad prelaunch baseline and before-window observation. There was
no used-swap growth, no swap-out and no GLM cgroup swap. Attribution is unknown.
Those attempts and their formulas remain unchanged.

Run this explicit external diagnostic in a fresh small zero-swap cgroup. It is
not imported by a server, is absent from production profiles, and adds no disabled
serving-path work. First capture an unloaded control for120seconds, then a separate
bounded observation spanning freeze, preparation and the unchanged512/128 replay.
Retain explicit phase timestamps around those existing actions. Observe only
PID/start ticks/name, UID, major/minor fault counters, VmSwap and cgroup membership;
never read other processes' command lines, environments or memory contents.

Bracket each1Hz census with global memory/swap counters, retain read errors,
process exits and identity changes, and record observer code/binary/boot identity.
Matching major-fault deltas narrow candidates but cannot prove swap causation:
file faults, shared mappings, readahead and disappearing tasks are limitations.
An event before model creation can narrow the phase without attributing it to GLM.
No diagnostic correlation replaces a failed qualification gate or a frozen replay.
The observer has a900second internal maximum and an external timeout; no privileges
or serving configuration changes are needed.
