# Optional named profile integration

Acceptance fixed before implementation: both experimental profiles render the
exact four-slot geometry and containment; the production profile and launch
parameter overrides are rejected; changed/unlisted runtime files and stale
process identities fail closed. All tests must pass. These are lifecycle tests,
not model qualification or performance evidence.

`red.log` is the genuine unchanged-code result (four errors, missing profiles
and module) at 721041dd. Existing production admission/default tests must also
continue to pass. Start stays explicit, uses the existing hardened wrapper,
and never writes the production switch's active/default state.
