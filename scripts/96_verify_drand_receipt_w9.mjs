#!/usr/local/libexec/glm52-w9/node

import { bls12_381 as bls } from "/usr/local/libexec/glm52-w9/noble/curves/bls12-381.js";
import { sha256 } from "/usr/local/libexec/glm52-w9/noble/hashes/sha2.js";

const PUBLIC_KEY = "868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31";
function fail(message) { process.stderr.write(`${message}\n`); process.exit(2); }
if (process.argv.length !== 6) fail("usage: verifier ROUND RANDOMNESS SIGNATURE PREVIOUS_SIGNATURE");
const [roundRaw, randomness, signature, previous] = process.argv.slice(2);
if (!/^[1-9][0-9]*$/.test(roundRaw) || !/^[0-9a-f]{64}$/.test(randomness) ||
    !/^[0-9a-f]{192}$/.test(signature) || !/^[0-9a-f]{192}$/.test(previous)) fail("invalid receipt encoding");
const round = BigInt(roundRaw); if (round > 0xffffffffffffffffn) fail("round exceeds uint64");
const roundBytes = new Uint8Array(8); new DataView(roundBytes.buffer).setBigUint64(0, round, false);
const previousBytes = Uint8Array.from(Buffer.from(previous, "hex"));
const input = new Uint8Array(previousBytes.length + roundBytes.length);
input.set(previousBytes); input.set(roundBytes, previousBytes.length);
const message = sha256(input);
if (Buffer.from(sha256(Uint8Array.from(Buffer.from(signature, "hex")))).toString("hex") !== randomness) fail("randomness mismatch");
let verified = false; try { verified = await bls.verify(signature, message, PUBLIC_KEY); } catch (error) { fail(`BLS error: ${error.message}`); }
if (!verified) fail("BLS signature failed");
process.stdout.write("DRAND_BLS_RECEIPT_OK\n");
