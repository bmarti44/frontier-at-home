# Candidate 6: never frozen, never run (`NO_GO` before freeze)

This directory holds the candidate 6 draft:
- `PREREG.md`: images on at 1M, gate clause 9 (MMMU ≥ 64/100), and the
  guarded holdout spend;
- the driver, the scorer and the mutation check (29/29).

None of it was committed as a freeze, and no start ran.

The pre-freeze diagnostic (`../diag-vision-2026-10-11/RESULT.md`) showed that
images and the direct 1M fill do not fit above the 10 GiB floor with a safe
margin. Run 2 stopped at 10.37 GiB, 16 s into the fill, after 100 images. The
diagnostic also scored MMMU-val-100 at 63/100 in chat mode, against the
default's 64. So this candidate is `NO_GO` and stays as the record of the plan.
