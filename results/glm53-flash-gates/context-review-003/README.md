# Final bounded context client correction

The module explicitly rejects optimized Python before importing its scorer,
so-O/-OO cannot disable its acceptance assertions. Fresh subprocess tests
cover both modes. A local pre-submission mutation also showed that nonstandard
NaN in an unused raw field was accepted by Python's default JSON parser. The
existing strict decoder now handles all response/metrics JSON, closing that
malformed-row case without a new parser framework.

All15 focused synthetic CPU tests pass. The prior native SSE grammar/token
coverage observation remains unchanged. Earlier independent H1/H2 findings are
closed; this candidate changes only the interpreter guard and strict JSON use.
This is test validation, not a full-context or model qualification result.
