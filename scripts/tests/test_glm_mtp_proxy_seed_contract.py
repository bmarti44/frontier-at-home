#!/usr/bin/env python3
import hashlib
import struct
import unittest


DOMAIN = b"R05-BLINDED-REAL-V2\x00"
EXPECTED = "84103b930e97824f02bc39736bd684595367e2f07a4fc22f5a1d8ca948558daa"
FIELDS = {
    "event_id": "12345678901",
    "ref": "refs/heads/glm52-rung0-io-submission",
    "created_at": "2026-08-06T19:15:00Z",
    "commit": "000102030405060708090a0b0c0d0e0f10111213",
    "manifest": "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
    "drand": "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
}


def selector_seed(fields: dict[str, str]) -> str:
    payload = bytearray(DOMAIN)
    for name in ("event_id", "ref", "created_at"):
        value = fields[name].encode("utf-8")
        payload += struct.pack("<I", len(value)) + value
    payload += bytes.fromhex(fields["commit"])
    payload += bytes.fromhex(fields["manifest"])
    payload += bytes.fromhex(fields["drand"])
    return hashlib.sha256(payload).hexdigest()


def scorer_seed(fields: dict[str, str]) -> str:
    framed = []
    for name in ("event_id", "ref", "created_at"):
        value = fields[name].encode("utf-8")
        framed.append(len(value).to_bytes(4, "little"))
        framed.append(value)
    raw = DOMAIN + b"".join(framed)
    raw += bytes.fromhex(fields["commit"])
    raw += bytes.fromhex(fields["manifest"])
    raw += bytes.fromhex(fields["drand"])
    return hashlib.sha256(raw).hexdigest()


class SeedContractTest(unittest.TestCase):
    def test_independent_implementations_match_fixed_vector(self) -> None:
        self.assertEqual(selector_seed(FIELDS), EXPECTED)
        self.assertEqual(scorer_seed(FIELDS), EXPECTED)

    def test_every_identity_field_is_bound(self) -> None:
        for name in ("event_id", "ref", "created_at", "commit", "manifest", "drand"):
            changed = dict(FIELDS)
            value = changed[name]
            changed[name] = ("1" if value[0] != "1" else "2") + value[1:]
            self.assertNotEqual(selector_seed(changed), EXPECTED, name)
            self.assertNotEqual(scorer_seed(changed), EXPECTED, name)

    def test_hex_text_is_not_raw_digest_serialization(self) -> None:
        payload = bytearray(DOMAIN)
        for name in ("event_id", "ref", "created_at"):
            value = FIELDS[name].encode()
            payload += struct.pack("<I", len(value)) + value
        payload += FIELDS["commit"].encode()
        payload += FIELDS["manifest"].encode()
        payload += FIELDS["drand"].encode()
        self.assertNotEqual(hashlib.sha256(payload).hexdigest(), EXPECTED)

    def test_order_endian_domain_and_trailing_mutations_differ(self) -> None:
        values = [FIELDS[name].encode() for name in ("event_id", "ref", "created_at")]
        reordered = DOMAIN + b"".join(
            struct.pack("<I", len(value)) + value for value in reversed(values)
        )
        big_endian = DOMAIN + b"".join(
            struct.pack(">I", len(value)) + value for value in values
        )
        suffix = bytes.fromhex(FIELDS["commit"] + FIELDS["manifest"] + FIELDS["drand"])
        for payload in (reordered + suffix, big_endian + suffix, DOMAIN[:-1] + suffix,
                        DOMAIN + b"\x00" + suffix):
            self.assertNotEqual(hashlib.sha256(payload).hexdigest(), EXPECTED)


if __name__ == "__main__":
    unittest.main()
