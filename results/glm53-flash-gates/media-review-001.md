# Media gate review

Candidate 1: `25cf7a9a`. Campaign review round 2. Both persistent reviewers
independently report high finding H1; no runtime launch occurred.

H1: policy counts declared content `type`, while pinned vLLM's
`_parse_chat_message_content_mm_part` honors direct media keys whenever `uuid`
is present. Pydantic preserves these extra keys. A text part containing UUID
and video URL passes admission but is interpreted as video. Four images plus
the disguised video bypass the intended image-or-video memory reservation.

Acceptance for the focused fix: reject UUID, undeclared per-kind keys, and
cross-type payloads before engine dispatch; retain valid text, tools, images,
and video unchanged. Include disguised image/video/embedding parts and text,
refusal, and cross-type combinations. Keep the actual frame-cap work separate.
