#!/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node
import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
  get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
}) : x)(function(x) {
  if (typeof require !== "undefined") return require.apply(this, arguments);
  throw Error('Dynamic require of "' + x + '" is not supported');
});
var __commonJS = (cb, mod) => function __require2() {
  return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/cryptoNode.js
var require_cryptoNode = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/cryptoNode.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.crypto = void 0;
    var nc = __require("node:crypto");
    exports.crypto = nc && typeof nc === "object" && "webcrypto" in nc ? nc.webcrypto : nc && typeof nc === "object" && "randomBytes" in nc ? nc : void 0;
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/utils.js
var require_utils = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/utils.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.wrapXOFConstructorWithOpts = exports.wrapConstructorWithOpts = exports.wrapConstructor = exports.Hash = exports.nextTick = exports.swap32IfBE = exports.byteSwapIfBE = exports.swap8IfBE = exports.isLE = void 0;
    exports.isBytes = isBytes;
    exports.anumber = anumber;
    exports.abytes = abytes;
    exports.ahash = ahash;
    exports.aexists = aexists;
    exports.aoutput = aoutput;
    exports.u8 = u8;
    exports.u32 = u32;
    exports.clean = clean;
    exports.createView = createView;
    exports.rotr = rotr;
    exports.rotl = rotl;
    exports.byteSwap = byteSwap;
    exports.byteSwap32 = byteSwap32;
    exports.bytesToHex = bytesToHex;
    exports.hexToBytes = hexToBytes;
    exports.asyncLoop = asyncLoop;
    exports.utf8ToBytes = utf8ToBytes;
    exports.bytesToUtf8 = bytesToUtf8;
    exports.toBytes = toBytes;
    exports.kdfInputToBytes = kdfInputToBytes;
    exports.concatBytes = concatBytes;
    exports.checkOpts = checkOpts;
    exports.createHasher = createHasher;
    exports.createOptHasher = createOptHasher;
    exports.createXOFer = createXOFer;
    exports.randomBytes = randomBytes;
    var crypto_1 = require_cryptoNode();
    function isBytes(a) {
      return a instanceof Uint8Array || ArrayBuffer.isView(a) && a.constructor.name === "Uint8Array";
    }
    function anumber(n) {
      if (!Number.isSafeInteger(n) || n < 0)
        throw new Error("positive integer expected, got " + n);
    }
    function abytes(b, ...lengths) {
      if (!isBytes(b))
        throw new Error("Uint8Array expected");
      if (lengths.length > 0 && !lengths.includes(b.length))
        throw new Error("Uint8Array expected of length " + lengths + ", got length=" + b.length);
    }
    function ahash(h) {
      if (typeof h !== "function" || typeof h.create !== "function")
        throw new Error("Hash should be wrapped by utils.createHasher");
      anumber(h.outputLen);
      anumber(h.blockLen);
    }
    function aexists(instance, checkFinished = true) {
      if (instance.destroyed)
        throw new Error("Hash instance has been destroyed");
      if (checkFinished && instance.finished)
        throw new Error("Hash#digest() has already been called");
    }
    function aoutput(out, instance) {
      abytes(out);
      const min = instance.outputLen;
      if (out.length < min) {
        throw new Error("digestInto() expects output buffer of length at least " + min);
      }
    }
    function u8(arr) {
      return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    }
    function u32(arr) {
      return new Uint32Array(arr.buffer, arr.byteOffset, Math.floor(arr.byteLength / 4));
    }
    function clean(...arrays) {
      for (let i = 0; i < arrays.length; i++) {
        arrays[i].fill(0);
      }
    }
    function createView(arr) {
      return new DataView(arr.buffer, arr.byteOffset, arr.byteLength);
    }
    function rotr(word, shift) {
      return word << 32 - shift | word >>> shift;
    }
    function rotl(word, shift) {
      return word << shift | word >>> 32 - shift >>> 0;
    }
    exports.isLE = (() => new Uint8Array(new Uint32Array([287454020]).buffer)[0] === 68)();
    function byteSwap(word) {
      return word << 24 & 4278190080 | word << 8 & 16711680 | word >>> 8 & 65280 | word >>> 24 & 255;
    }
    exports.swap8IfBE = exports.isLE ? (n) => n : (n) => byteSwap(n);
    exports.byteSwapIfBE = exports.swap8IfBE;
    function byteSwap32(arr) {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = byteSwap(arr[i]);
      }
      return arr;
    }
    exports.swap32IfBE = exports.isLE ? (u) => u : byteSwap32;
    var hasHexBuiltin = /* @__PURE__ */ (() => (
      // @ts-ignore
      typeof Uint8Array.from([]).toHex === "function" && typeof Uint8Array.fromHex === "function"
    ))();
    var hexes = /* @__PURE__ */ Array.from({ length: 256 }, (_, i) => i.toString(16).padStart(2, "0"));
    function bytesToHex(bytes) {
      abytes(bytes);
      if (hasHexBuiltin)
        return bytes.toHex();
      let hex = "";
      for (let i = 0; i < bytes.length; i++) {
        hex += hexes[bytes[i]];
      }
      return hex;
    }
    var asciis = { _0: 48, _9: 57, A: 65, F: 70, a: 97, f: 102 };
    function asciiToBase16(ch) {
      if (ch >= asciis._0 && ch <= asciis._9)
        return ch - asciis._0;
      if (ch >= asciis.A && ch <= asciis.F)
        return ch - (asciis.A - 10);
      if (ch >= asciis.a && ch <= asciis.f)
        return ch - (asciis.a - 10);
      return;
    }
    function hexToBytes(hex) {
      if (typeof hex !== "string")
        throw new Error("hex string expected, got " + typeof hex);
      if (hasHexBuiltin)
        return Uint8Array.fromHex(hex);
      const hl = hex.length;
      const al = hl / 2;
      if (hl % 2)
        throw new Error("hex string expected, got unpadded hex of length " + hl);
      const array = new Uint8Array(al);
      for (let ai = 0, hi = 0; ai < al; ai++, hi += 2) {
        const n1 = asciiToBase16(hex.charCodeAt(hi));
        const n2 = asciiToBase16(hex.charCodeAt(hi + 1));
        if (n1 === void 0 || n2 === void 0) {
          const char = hex[hi] + hex[hi + 1];
          throw new Error('hex string expected, got non-hex character "' + char + '" at index ' + hi);
        }
        array[ai] = n1 * 16 + n2;
      }
      return array;
    }
    var nextTick = async () => {
    };
    exports.nextTick = nextTick;
    async function asyncLoop(iters, tick, cb) {
      let ts = Date.now();
      for (let i = 0; i < iters; i++) {
        cb(i);
        const diff = Date.now() - ts;
        if (diff >= 0 && diff < tick)
          continue;
        await (0, exports.nextTick)();
        ts += diff;
      }
    }
    function utf8ToBytes(str) {
      if (typeof str !== "string")
        throw new Error("string expected");
      return new Uint8Array(new TextEncoder().encode(str));
    }
    function bytesToUtf8(bytes) {
      return new TextDecoder().decode(bytes);
    }
    function toBytes(data) {
      if (typeof data === "string")
        data = utf8ToBytes(data);
      abytes(data);
      return data;
    }
    function kdfInputToBytes(data) {
      if (typeof data === "string")
        data = utf8ToBytes(data);
      abytes(data);
      return data;
    }
    function concatBytes(...arrays) {
      let sum = 0;
      for (let i = 0; i < arrays.length; i++) {
        const a = arrays[i];
        abytes(a);
        sum += a.length;
      }
      const res = new Uint8Array(sum);
      for (let i = 0, pad = 0; i < arrays.length; i++) {
        const a = arrays[i];
        res.set(a, pad);
        pad += a.length;
      }
      return res;
    }
    function checkOpts(defaults, opts) {
      if (opts !== void 0 && {}.toString.call(opts) !== "[object Object]")
        throw new Error("options should be object or undefined");
      const merged = Object.assign(defaults, opts);
      return merged;
    }
    var Hash = class {
    };
    exports.Hash = Hash;
    function createHasher(hashCons) {
      const hashC = (msg) => hashCons().update(toBytes(msg)).digest();
      const tmp = hashCons();
      hashC.outputLen = tmp.outputLen;
      hashC.blockLen = tmp.blockLen;
      hashC.create = () => hashCons();
      return hashC;
    }
    function createOptHasher(hashCons) {
      const hashC = (msg, opts) => hashCons(opts).update(toBytes(msg)).digest();
      const tmp = hashCons({});
      hashC.outputLen = tmp.outputLen;
      hashC.blockLen = tmp.blockLen;
      hashC.create = (opts) => hashCons(opts);
      return hashC;
    }
    function createXOFer(hashCons) {
      const hashC = (msg, opts) => hashCons(opts).update(toBytes(msg)).digest();
      const tmp = hashCons({});
      hashC.outputLen = tmp.outputLen;
      hashC.blockLen = tmp.blockLen;
      hashC.create = (opts) => hashCons(opts);
      return hashC;
    }
    exports.wrapConstructor = createHasher;
    exports.wrapConstructorWithOpts = createOptHasher;
    exports.wrapXOFConstructorWithOpts = createXOFer;
    function randomBytes(bytesLength = 32) {
      if (crypto_1.crypto && typeof crypto_1.crypto.getRandomValues === "function") {
        return crypto_1.crypto.getRandomValues(new Uint8Array(bytesLength));
      }
      if (crypto_1.crypto && typeof crypto_1.crypto.randomBytes === "function") {
        return Uint8Array.from(crypto_1.crypto.randomBytes(bytesLength));
      }
      throw new Error("crypto.getRandomValues must be defined");
    }
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/_md.js
var require_md = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/_md.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.SHA512_IV = exports.SHA384_IV = exports.SHA224_IV = exports.SHA256_IV = exports.HashMD = void 0;
    exports.setBigUint64 = setBigUint64;
    exports.Chi = Chi;
    exports.Maj = Maj;
    var utils_ts_1 = require_utils();
    function setBigUint64(view, byteOffset, value, isLE) {
      if (typeof view.setBigUint64 === "function")
        return view.setBigUint64(byteOffset, value, isLE);
      const _32n = BigInt(32);
      const _u32_max = BigInt(4294967295);
      const wh = Number(value >> _32n & _u32_max);
      const wl = Number(value & _u32_max);
      const h = isLE ? 4 : 0;
      const l = isLE ? 0 : 4;
      view.setUint32(byteOffset + h, wh, isLE);
      view.setUint32(byteOffset + l, wl, isLE);
    }
    function Chi(a, b, c) {
      return a & b ^ ~a & c;
    }
    function Maj(a, b, c) {
      return a & b ^ a & c ^ b & c;
    }
    var HashMD = class extends utils_ts_1.Hash {
      constructor(blockLen, outputLen, padOffset, isLE) {
        super();
        this.finished = false;
        this.length = 0;
        this.pos = 0;
        this.destroyed = false;
        this.blockLen = blockLen;
        this.outputLen = outputLen;
        this.padOffset = padOffset;
        this.isLE = isLE;
        this.buffer = new Uint8Array(blockLen);
        this.view = (0, utils_ts_1.createView)(this.buffer);
      }
      update(data) {
        (0, utils_ts_1.aexists)(this);
        data = (0, utils_ts_1.toBytes)(data);
        (0, utils_ts_1.abytes)(data);
        const { view, buffer, blockLen } = this;
        const len = data.length;
        for (let pos = 0; pos < len; ) {
          const take = Math.min(blockLen - this.pos, len - pos);
          if (take === blockLen) {
            const dataView = (0, utils_ts_1.createView)(data);
            for (; blockLen <= len - pos; pos += blockLen)
              this.process(dataView, pos);
            continue;
          }
          buffer.set(data.subarray(pos, pos + take), this.pos);
          this.pos += take;
          pos += take;
          if (this.pos === blockLen) {
            this.process(view, 0);
            this.pos = 0;
          }
        }
        this.length += data.length;
        this.roundClean();
        return this;
      }
      digestInto(out) {
        (0, utils_ts_1.aexists)(this);
        (0, utils_ts_1.aoutput)(out, this);
        this.finished = true;
        const { buffer, view, blockLen, isLE } = this;
        let { pos } = this;
        buffer[pos++] = 128;
        (0, utils_ts_1.clean)(this.buffer.subarray(pos));
        if (this.padOffset > blockLen - pos) {
          this.process(view, 0);
          pos = 0;
        }
        for (let i = pos; i < blockLen; i++)
          buffer[i] = 0;
        setBigUint64(view, blockLen - 8, BigInt(this.length * 8), isLE);
        this.process(view, 0);
        const oview = (0, utils_ts_1.createView)(out);
        const len = this.outputLen;
        if (len % 4)
          throw new Error("_sha2: outputLen should be aligned to 32bit");
        const outLen = len / 4;
        const state = this.get();
        if (outLen > state.length)
          throw new Error("_sha2: outputLen bigger than state");
        for (let i = 0; i < outLen; i++)
          oview.setUint32(4 * i, state[i], isLE);
      }
      digest() {
        const { buffer, outputLen } = this;
        this.digestInto(buffer);
        const res = buffer.slice(0, outputLen);
        this.destroy();
        return res;
      }
      _cloneInto(to) {
        to || (to = new this.constructor());
        to.set(...this.get());
        const { blockLen, buffer, length, finished, destroyed, pos } = this;
        to.destroyed = destroyed;
        to.finished = finished;
        to.length = length;
        to.pos = pos;
        if (length % blockLen)
          to.buffer.set(buffer);
        return to;
      }
      clone() {
        return this._cloneInto();
      }
    };
    exports.HashMD = HashMD;
    exports.SHA256_IV = Uint32Array.from([
      1779033703,
      3144134277,
      1013904242,
      2773480762,
      1359893119,
      2600822924,
      528734635,
      1541459225
    ]);
    exports.SHA224_IV = Uint32Array.from([
      3238371032,
      914150663,
      812702999,
      4144912697,
      4290775857,
      1750603025,
      1694076839,
      3204075428
    ]);
    exports.SHA384_IV = Uint32Array.from([
      3418070365,
      3238371032,
      1654270250,
      914150663,
      2438529370,
      812702999,
      355462360,
      4144912697,
      1731405415,
      4290775857,
      2394180231,
      1750603025,
      3675008525,
      1694076839,
      1203062813,
      3204075428
    ]);
    exports.SHA512_IV = Uint32Array.from([
      1779033703,
      4089235720,
      3144134277,
      2227873595,
      1013904242,
      4271175723,
      2773480762,
      1595750129,
      1359893119,
      2917565137,
      2600822924,
      725511199,
      528734635,
      4215389547,
      1541459225,
      327033209
    ]);
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/_u64.js
var require_u64 = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/_u64.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.toBig = exports.shrSL = exports.shrSH = exports.rotrSL = exports.rotrSH = exports.rotrBL = exports.rotrBH = exports.rotr32L = exports.rotr32H = exports.rotlSL = exports.rotlSH = exports.rotlBL = exports.rotlBH = exports.add5L = exports.add5H = exports.add4L = exports.add4H = exports.add3L = exports.add3H = void 0;
    exports.add = add;
    exports.fromBig = fromBig;
    exports.split = split;
    var U32_MASK64 = /* @__PURE__ */ BigInt(2 ** 32 - 1);
    var _32n = /* @__PURE__ */ BigInt(32);
    function fromBig(n, le = false) {
      if (le)
        return { h: Number(n & U32_MASK64), l: Number(n >> _32n & U32_MASK64) };
      return { h: Number(n >> _32n & U32_MASK64) | 0, l: Number(n & U32_MASK64) | 0 };
    }
    function split(lst, le = false) {
      const len = lst.length;
      let Ah = new Uint32Array(len);
      let Al = new Uint32Array(len);
      for (let i = 0; i < len; i++) {
        const { h, l } = fromBig(lst[i], le);
        [Ah[i], Al[i]] = [h, l];
      }
      return [Ah, Al];
    }
    var toBig = (h, l) => BigInt(h >>> 0) << _32n | BigInt(l >>> 0);
    exports.toBig = toBig;
    var shrSH = (h, _l, s) => h >>> s;
    exports.shrSH = shrSH;
    var shrSL = (h, l, s) => h << 32 - s | l >>> s;
    exports.shrSL = shrSL;
    var rotrSH = (h, l, s) => h >>> s | l << 32 - s;
    exports.rotrSH = rotrSH;
    var rotrSL = (h, l, s) => h << 32 - s | l >>> s;
    exports.rotrSL = rotrSL;
    var rotrBH = (h, l, s) => h << 64 - s | l >>> s - 32;
    exports.rotrBH = rotrBH;
    var rotrBL = (h, l, s) => h >>> s - 32 | l << 64 - s;
    exports.rotrBL = rotrBL;
    var rotr32H = (_h, l) => l;
    exports.rotr32H = rotr32H;
    var rotr32L = (h, _l) => h;
    exports.rotr32L = rotr32L;
    var rotlSH = (h, l, s) => h << s | l >>> 32 - s;
    exports.rotlSH = rotlSH;
    var rotlSL = (h, l, s) => l << s | h >>> 32 - s;
    exports.rotlSL = rotlSL;
    var rotlBH = (h, l, s) => l << s - 32 | h >>> 64 - s;
    exports.rotlBH = rotlBH;
    var rotlBL = (h, l, s) => h << s - 32 | l >>> 64 - s;
    exports.rotlBL = rotlBL;
    function add(Ah, Al, Bh, Bl) {
      const l = (Al >>> 0) + (Bl >>> 0);
      return { h: Ah + Bh + (l / 2 ** 32 | 0) | 0, l: l | 0 };
    }
    var add3L = (Al, Bl, Cl) => (Al >>> 0) + (Bl >>> 0) + (Cl >>> 0);
    exports.add3L = add3L;
    var add3H = (low, Ah, Bh, Ch) => Ah + Bh + Ch + (low / 2 ** 32 | 0) | 0;
    exports.add3H = add3H;
    var add4L = (Al, Bl, Cl, Dl) => (Al >>> 0) + (Bl >>> 0) + (Cl >>> 0) + (Dl >>> 0);
    exports.add4L = add4L;
    var add4H = (low, Ah, Bh, Ch, Dh) => Ah + Bh + Ch + Dh + (low / 2 ** 32 | 0) | 0;
    exports.add4H = add4H;
    var add5L = (Al, Bl, Cl, Dl, El) => (Al >>> 0) + (Bl >>> 0) + (Cl >>> 0) + (Dl >>> 0) + (El >>> 0);
    exports.add5L = add5L;
    var add5H = (low, Ah, Bh, Ch, Dh, Eh) => Ah + Bh + Ch + Dh + Eh + (low / 2 ** 32 | 0) | 0;
    exports.add5H = add5H;
    var u64 = {
      fromBig,
      split,
      toBig,
      shrSH,
      shrSL,
      rotrSH,
      rotrSL,
      rotrBH,
      rotrBL,
      rotr32H,
      rotr32L,
      rotlSH,
      rotlSL,
      rotlBH,
      rotlBL,
      add,
      add3L,
      add3H,
      add4L,
      add4H,
      add5H,
      add5L
    };
    exports.default = u64;
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/sha2.js
var require_sha2 = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/sha2.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.sha512_224 = exports.sha512_256 = exports.sha384 = exports.sha512 = exports.sha224 = exports.sha256 = exports.SHA512_256 = exports.SHA512_224 = exports.SHA384 = exports.SHA512 = exports.SHA224 = exports.SHA256 = void 0;
    var _md_ts_1 = require_md();
    var u64 = require_u64();
    var utils_ts_1 = require_utils();
    var SHA256_K = /* @__PURE__ */ Uint32Array.from([
      1116352408,
      1899447441,
      3049323471,
      3921009573,
      961987163,
      1508970993,
      2453635748,
      2870763221,
      3624381080,
      310598401,
      607225278,
      1426881987,
      1925078388,
      2162078206,
      2614888103,
      3248222580,
      3835390401,
      4022224774,
      264347078,
      604807628,
      770255983,
      1249150122,
      1555081692,
      1996064986,
      2554220882,
      2821834349,
      2952996808,
      3210313671,
      3336571891,
      3584528711,
      113926993,
      338241895,
      666307205,
      773529912,
      1294757372,
      1396182291,
      1695183700,
      1986661051,
      2177026350,
      2456956037,
      2730485921,
      2820302411,
      3259730800,
      3345764771,
      3516065817,
      3600352804,
      4094571909,
      275423344,
      430227734,
      506948616,
      659060556,
      883997877,
      958139571,
      1322822218,
      1537002063,
      1747873779,
      1955562222,
      2024104815,
      2227730452,
      2361852424,
      2428436474,
      2756734187,
      3204031479,
      3329325298
    ]);
    var SHA256_W = /* @__PURE__ */ new Uint32Array(64);
    var SHA256 = class extends _md_ts_1.HashMD {
      constructor(outputLen = 32) {
        super(64, outputLen, 8, false);
        this.A = _md_ts_1.SHA256_IV[0] | 0;
        this.B = _md_ts_1.SHA256_IV[1] | 0;
        this.C = _md_ts_1.SHA256_IV[2] | 0;
        this.D = _md_ts_1.SHA256_IV[3] | 0;
        this.E = _md_ts_1.SHA256_IV[4] | 0;
        this.F = _md_ts_1.SHA256_IV[5] | 0;
        this.G = _md_ts_1.SHA256_IV[6] | 0;
        this.H = _md_ts_1.SHA256_IV[7] | 0;
      }
      get() {
        const { A, B, C, D, E, F, G, H } = this;
        return [A, B, C, D, E, F, G, H];
      }
      // prettier-ignore
      set(A, B, C, D, E, F, G, H) {
        this.A = A | 0;
        this.B = B | 0;
        this.C = C | 0;
        this.D = D | 0;
        this.E = E | 0;
        this.F = F | 0;
        this.G = G | 0;
        this.H = H | 0;
      }
      process(view, offset) {
        for (let i = 0; i < 16; i++, offset += 4)
          SHA256_W[i] = view.getUint32(offset, false);
        for (let i = 16; i < 64; i++) {
          const W15 = SHA256_W[i - 15];
          const W2 = SHA256_W[i - 2];
          const s0 = (0, utils_ts_1.rotr)(W15, 7) ^ (0, utils_ts_1.rotr)(W15, 18) ^ W15 >>> 3;
          const s1 = (0, utils_ts_1.rotr)(W2, 17) ^ (0, utils_ts_1.rotr)(W2, 19) ^ W2 >>> 10;
          SHA256_W[i] = s1 + SHA256_W[i - 7] + s0 + SHA256_W[i - 16] | 0;
        }
        let { A, B, C, D, E, F, G, H } = this;
        for (let i = 0; i < 64; i++) {
          const sigma1 = (0, utils_ts_1.rotr)(E, 6) ^ (0, utils_ts_1.rotr)(E, 11) ^ (0, utils_ts_1.rotr)(E, 25);
          const T1 = H + sigma1 + (0, _md_ts_1.Chi)(E, F, G) + SHA256_K[i] + SHA256_W[i] | 0;
          const sigma0 = (0, utils_ts_1.rotr)(A, 2) ^ (0, utils_ts_1.rotr)(A, 13) ^ (0, utils_ts_1.rotr)(A, 22);
          const T2 = sigma0 + (0, _md_ts_1.Maj)(A, B, C) | 0;
          H = G;
          G = F;
          F = E;
          E = D + T1 | 0;
          D = C;
          C = B;
          B = A;
          A = T1 + T2 | 0;
        }
        A = A + this.A | 0;
        B = B + this.B | 0;
        C = C + this.C | 0;
        D = D + this.D | 0;
        E = E + this.E | 0;
        F = F + this.F | 0;
        G = G + this.G | 0;
        H = H + this.H | 0;
        this.set(A, B, C, D, E, F, G, H);
      }
      roundClean() {
        (0, utils_ts_1.clean)(SHA256_W);
      }
      destroy() {
        this.set(0, 0, 0, 0, 0, 0, 0, 0);
        (0, utils_ts_1.clean)(this.buffer);
      }
    };
    exports.SHA256 = SHA256;
    var SHA224 = class extends SHA256 {
      constructor() {
        super(28);
        this.A = _md_ts_1.SHA224_IV[0] | 0;
        this.B = _md_ts_1.SHA224_IV[1] | 0;
        this.C = _md_ts_1.SHA224_IV[2] | 0;
        this.D = _md_ts_1.SHA224_IV[3] | 0;
        this.E = _md_ts_1.SHA224_IV[4] | 0;
        this.F = _md_ts_1.SHA224_IV[5] | 0;
        this.G = _md_ts_1.SHA224_IV[6] | 0;
        this.H = _md_ts_1.SHA224_IV[7] | 0;
      }
    };
    exports.SHA224 = SHA224;
    var K512 = /* @__PURE__ */ (() => u64.split([
      "0x428a2f98d728ae22",
      "0x7137449123ef65cd",
      "0xb5c0fbcfec4d3b2f",
      "0xe9b5dba58189dbbc",
      "0x3956c25bf348b538",
      "0x59f111f1b605d019",
      "0x923f82a4af194f9b",
      "0xab1c5ed5da6d8118",
      "0xd807aa98a3030242",
      "0x12835b0145706fbe",
      "0x243185be4ee4b28c",
      "0x550c7dc3d5ffb4e2",
      "0x72be5d74f27b896f",
      "0x80deb1fe3b1696b1",
      "0x9bdc06a725c71235",
      "0xc19bf174cf692694",
      "0xe49b69c19ef14ad2",
      "0xefbe4786384f25e3",
      "0x0fc19dc68b8cd5b5",
      "0x240ca1cc77ac9c65",
      "0x2de92c6f592b0275",
      "0x4a7484aa6ea6e483",
      "0x5cb0a9dcbd41fbd4",
      "0x76f988da831153b5",
      "0x983e5152ee66dfab",
      "0xa831c66d2db43210",
      "0xb00327c898fb213f",
      "0xbf597fc7beef0ee4",
      "0xc6e00bf33da88fc2",
      "0xd5a79147930aa725",
      "0x06ca6351e003826f",
      "0x142929670a0e6e70",
      "0x27b70a8546d22ffc",
      "0x2e1b21385c26c926",
      "0x4d2c6dfc5ac42aed",
      "0x53380d139d95b3df",
      "0x650a73548baf63de",
      "0x766a0abb3c77b2a8",
      "0x81c2c92e47edaee6",
      "0x92722c851482353b",
      "0xa2bfe8a14cf10364",
      "0xa81a664bbc423001",
      "0xc24b8b70d0f89791",
      "0xc76c51a30654be30",
      "0xd192e819d6ef5218",
      "0xd69906245565a910",
      "0xf40e35855771202a",
      "0x106aa07032bbd1b8",
      "0x19a4c116b8d2d0c8",
      "0x1e376c085141ab53",
      "0x2748774cdf8eeb99",
      "0x34b0bcb5e19b48a8",
      "0x391c0cb3c5c95a63",
      "0x4ed8aa4ae3418acb",
      "0x5b9cca4f7763e373",
      "0x682e6ff3d6b2b8a3",
      "0x748f82ee5defb2fc",
      "0x78a5636f43172f60",
      "0x84c87814a1f0ab72",
      "0x8cc702081a6439ec",
      "0x90befffa23631e28",
      "0xa4506cebde82bde9",
      "0xbef9a3f7b2c67915",
      "0xc67178f2e372532b",
      "0xca273eceea26619c",
      "0xd186b8c721c0c207",
      "0xeada7dd6cde0eb1e",
      "0xf57d4f7fee6ed178",
      "0x06f067aa72176fba",
      "0x0a637dc5a2c898a6",
      "0x113f9804bef90dae",
      "0x1b710b35131c471b",
      "0x28db77f523047d84",
      "0x32caab7b40c72493",
      "0x3c9ebe0a15c9bebc",
      "0x431d67c49c100d4c",
      "0x4cc5d4becb3e42b6",
      "0x597f299cfc657e2a",
      "0x5fcb6fab3ad6faec",
      "0x6c44198c4a475817"
    ].map((n) => BigInt(n))))();
    var SHA512_Kh = /* @__PURE__ */ (() => K512[0])();
    var SHA512_Kl = /* @__PURE__ */ (() => K512[1])();
    var SHA512_W_H = /* @__PURE__ */ new Uint32Array(80);
    var SHA512_W_L = /* @__PURE__ */ new Uint32Array(80);
    var SHA512 = class extends _md_ts_1.HashMD {
      constructor(outputLen = 64) {
        super(128, outputLen, 16, false);
        this.Ah = _md_ts_1.SHA512_IV[0] | 0;
        this.Al = _md_ts_1.SHA512_IV[1] | 0;
        this.Bh = _md_ts_1.SHA512_IV[2] | 0;
        this.Bl = _md_ts_1.SHA512_IV[3] | 0;
        this.Ch = _md_ts_1.SHA512_IV[4] | 0;
        this.Cl = _md_ts_1.SHA512_IV[5] | 0;
        this.Dh = _md_ts_1.SHA512_IV[6] | 0;
        this.Dl = _md_ts_1.SHA512_IV[7] | 0;
        this.Eh = _md_ts_1.SHA512_IV[8] | 0;
        this.El = _md_ts_1.SHA512_IV[9] | 0;
        this.Fh = _md_ts_1.SHA512_IV[10] | 0;
        this.Fl = _md_ts_1.SHA512_IV[11] | 0;
        this.Gh = _md_ts_1.SHA512_IV[12] | 0;
        this.Gl = _md_ts_1.SHA512_IV[13] | 0;
        this.Hh = _md_ts_1.SHA512_IV[14] | 0;
        this.Hl = _md_ts_1.SHA512_IV[15] | 0;
      }
      // prettier-ignore
      get() {
        const { Ah, Al, Bh, Bl, Ch, Cl, Dh, Dl, Eh, El, Fh, Fl, Gh, Gl, Hh, Hl } = this;
        return [Ah, Al, Bh, Bl, Ch, Cl, Dh, Dl, Eh, El, Fh, Fl, Gh, Gl, Hh, Hl];
      }
      // prettier-ignore
      set(Ah, Al, Bh, Bl, Ch, Cl, Dh, Dl, Eh, El, Fh, Fl, Gh, Gl, Hh, Hl) {
        this.Ah = Ah | 0;
        this.Al = Al | 0;
        this.Bh = Bh | 0;
        this.Bl = Bl | 0;
        this.Ch = Ch | 0;
        this.Cl = Cl | 0;
        this.Dh = Dh | 0;
        this.Dl = Dl | 0;
        this.Eh = Eh | 0;
        this.El = El | 0;
        this.Fh = Fh | 0;
        this.Fl = Fl | 0;
        this.Gh = Gh | 0;
        this.Gl = Gl | 0;
        this.Hh = Hh | 0;
        this.Hl = Hl | 0;
      }
      process(view, offset) {
        for (let i = 0; i < 16; i++, offset += 4) {
          SHA512_W_H[i] = view.getUint32(offset);
          SHA512_W_L[i] = view.getUint32(offset += 4);
        }
        for (let i = 16; i < 80; i++) {
          const W15h = SHA512_W_H[i - 15] | 0;
          const W15l = SHA512_W_L[i - 15] | 0;
          const s0h = u64.rotrSH(W15h, W15l, 1) ^ u64.rotrSH(W15h, W15l, 8) ^ u64.shrSH(W15h, W15l, 7);
          const s0l = u64.rotrSL(W15h, W15l, 1) ^ u64.rotrSL(W15h, W15l, 8) ^ u64.shrSL(W15h, W15l, 7);
          const W2h = SHA512_W_H[i - 2] | 0;
          const W2l = SHA512_W_L[i - 2] | 0;
          const s1h = u64.rotrSH(W2h, W2l, 19) ^ u64.rotrBH(W2h, W2l, 61) ^ u64.shrSH(W2h, W2l, 6);
          const s1l = u64.rotrSL(W2h, W2l, 19) ^ u64.rotrBL(W2h, W2l, 61) ^ u64.shrSL(W2h, W2l, 6);
          const SUMl = u64.add4L(s0l, s1l, SHA512_W_L[i - 7], SHA512_W_L[i - 16]);
          const SUMh = u64.add4H(SUMl, s0h, s1h, SHA512_W_H[i - 7], SHA512_W_H[i - 16]);
          SHA512_W_H[i] = SUMh | 0;
          SHA512_W_L[i] = SUMl | 0;
        }
        let { Ah, Al, Bh, Bl, Ch, Cl, Dh, Dl, Eh, El, Fh, Fl, Gh, Gl, Hh, Hl } = this;
        for (let i = 0; i < 80; i++) {
          const sigma1h = u64.rotrSH(Eh, El, 14) ^ u64.rotrSH(Eh, El, 18) ^ u64.rotrBH(Eh, El, 41);
          const sigma1l = u64.rotrSL(Eh, El, 14) ^ u64.rotrSL(Eh, El, 18) ^ u64.rotrBL(Eh, El, 41);
          const CHIh = Eh & Fh ^ ~Eh & Gh;
          const CHIl = El & Fl ^ ~El & Gl;
          const T1ll = u64.add5L(Hl, sigma1l, CHIl, SHA512_Kl[i], SHA512_W_L[i]);
          const T1h = u64.add5H(T1ll, Hh, sigma1h, CHIh, SHA512_Kh[i], SHA512_W_H[i]);
          const T1l = T1ll | 0;
          const sigma0h = u64.rotrSH(Ah, Al, 28) ^ u64.rotrBH(Ah, Al, 34) ^ u64.rotrBH(Ah, Al, 39);
          const sigma0l = u64.rotrSL(Ah, Al, 28) ^ u64.rotrBL(Ah, Al, 34) ^ u64.rotrBL(Ah, Al, 39);
          const MAJh = Ah & Bh ^ Ah & Ch ^ Bh & Ch;
          const MAJl = Al & Bl ^ Al & Cl ^ Bl & Cl;
          Hh = Gh | 0;
          Hl = Gl | 0;
          Gh = Fh | 0;
          Gl = Fl | 0;
          Fh = Eh | 0;
          Fl = El | 0;
          ({ h: Eh, l: El } = u64.add(Dh | 0, Dl | 0, T1h | 0, T1l | 0));
          Dh = Ch | 0;
          Dl = Cl | 0;
          Ch = Bh | 0;
          Cl = Bl | 0;
          Bh = Ah | 0;
          Bl = Al | 0;
          const All = u64.add3L(T1l, sigma0l, MAJl);
          Ah = u64.add3H(All, T1h, sigma0h, MAJh);
          Al = All | 0;
        }
        ({ h: Ah, l: Al } = u64.add(this.Ah | 0, this.Al | 0, Ah | 0, Al | 0));
        ({ h: Bh, l: Bl } = u64.add(this.Bh | 0, this.Bl | 0, Bh | 0, Bl | 0));
        ({ h: Ch, l: Cl } = u64.add(this.Ch | 0, this.Cl | 0, Ch | 0, Cl | 0));
        ({ h: Dh, l: Dl } = u64.add(this.Dh | 0, this.Dl | 0, Dh | 0, Dl | 0));
        ({ h: Eh, l: El } = u64.add(this.Eh | 0, this.El | 0, Eh | 0, El | 0));
        ({ h: Fh, l: Fl } = u64.add(this.Fh | 0, this.Fl | 0, Fh | 0, Fl | 0));
        ({ h: Gh, l: Gl } = u64.add(this.Gh | 0, this.Gl | 0, Gh | 0, Gl | 0));
        ({ h: Hh, l: Hl } = u64.add(this.Hh | 0, this.Hl | 0, Hh | 0, Hl | 0));
        this.set(Ah, Al, Bh, Bl, Ch, Cl, Dh, Dl, Eh, El, Fh, Fl, Gh, Gl, Hh, Hl);
      }
      roundClean() {
        (0, utils_ts_1.clean)(SHA512_W_H, SHA512_W_L);
      }
      destroy() {
        (0, utils_ts_1.clean)(this.buffer);
        this.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      }
    };
    exports.SHA512 = SHA512;
    var SHA384 = class extends SHA512 {
      constructor() {
        super(48);
        this.Ah = _md_ts_1.SHA384_IV[0] | 0;
        this.Al = _md_ts_1.SHA384_IV[1] | 0;
        this.Bh = _md_ts_1.SHA384_IV[2] | 0;
        this.Bl = _md_ts_1.SHA384_IV[3] | 0;
        this.Ch = _md_ts_1.SHA384_IV[4] | 0;
        this.Cl = _md_ts_1.SHA384_IV[5] | 0;
        this.Dh = _md_ts_1.SHA384_IV[6] | 0;
        this.Dl = _md_ts_1.SHA384_IV[7] | 0;
        this.Eh = _md_ts_1.SHA384_IV[8] | 0;
        this.El = _md_ts_1.SHA384_IV[9] | 0;
        this.Fh = _md_ts_1.SHA384_IV[10] | 0;
        this.Fl = _md_ts_1.SHA384_IV[11] | 0;
        this.Gh = _md_ts_1.SHA384_IV[12] | 0;
        this.Gl = _md_ts_1.SHA384_IV[13] | 0;
        this.Hh = _md_ts_1.SHA384_IV[14] | 0;
        this.Hl = _md_ts_1.SHA384_IV[15] | 0;
      }
    };
    exports.SHA384 = SHA384;
    var T224_IV = /* @__PURE__ */ Uint32Array.from([
      2352822216,
      424955298,
      1944164710,
      2312950998,
      502970286,
      855612546,
      1738396948,
      1479516111,
      258812777,
      2077511080,
      2011393907,
      79989058,
      1067287976,
      1780299464,
      286451373,
      2446758561
    ]);
    var T256_IV = /* @__PURE__ */ Uint32Array.from([
      573645204,
      4230739756,
      2673172387,
      3360449730,
      596883563,
      1867755857,
      2520282905,
      1497426621,
      2519219938,
      2827943907,
      3193839141,
      1401305490,
      721525244,
      746961066,
      246885852,
      2177182882
    ]);
    var SHA512_224 = class extends SHA512 {
      constructor() {
        super(28);
        this.Ah = T224_IV[0] | 0;
        this.Al = T224_IV[1] | 0;
        this.Bh = T224_IV[2] | 0;
        this.Bl = T224_IV[3] | 0;
        this.Ch = T224_IV[4] | 0;
        this.Cl = T224_IV[5] | 0;
        this.Dh = T224_IV[6] | 0;
        this.Dl = T224_IV[7] | 0;
        this.Eh = T224_IV[8] | 0;
        this.El = T224_IV[9] | 0;
        this.Fh = T224_IV[10] | 0;
        this.Fl = T224_IV[11] | 0;
        this.Gh = T224_IV[12] | 0;
        this.Gl = T224_IV[13] | 0;
        this.Hh = T224_IV[14] | 0;
        this.Hl = T224_IV[15] | 0;
      }
    };
    exports.SHA512_224 = SHA512_224;
    var SHA512_256 = class extends SHA512 {
      constructor() {
        super(32);
        this.Ah = T256_IV[0] | 0;
        this.Al = T256_IV[1] | 0;
        this.Bh = T256_IV[2] | 0;
        this.Bl = T256_IV[3] | 0;
        this.Ch = T256_IV[4] | 0;
        this.Cl = T256_IV[5] | 0;
        this.Dh = T256_IV[6] | 0;
        this.Dl = T256_IV[7] | 0;
        this.Eh = T256_IV[8] | 0;
        this.El = T256_IV[9] | 0;
        this.Fh = T256_IV[10] | 0;
        this.Fl = T256_IV[11] | 0;
        this.Gh = T256_IV[12] | 0;
        this.Gl = T256_IV[13] | 0;
        this.Hh = T256_IV[14] | 0;
        this.Hl = T256_IV[15] | 0;
      }
    };
    exports.SHA512_256 = SHA512_256;
    exports.sha256 = (0, utils_ts_1.createHasher)(() => new SHA256());
    exports.sha224 = (0, utils_ts_1.createHasher)(() => new SHA224());
    exports.sha512 = (0, utils_ts_1.createHasher)(() => new SHA512());
    exports.sha384 = (0, utils_ts_1.createHasher)(() => new SHA384());
    exports.sha512_256 = (0, utils_ts_1.createHasher)(() => new SHA512_256());
    exports.sha512_224 = (0, utils_ts_1.createHasher)(() => new SHA512_224());
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/utils.js
var require_utils2 = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/utils.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.notImplemented = exports.bitMask = exports.utf8ToBytes = exports.randomBytes = exports.isBytes = exports.hexToBytes = exports.concatBytes = exports.bytesToUtf8 = exports.bytesToHex = exports.anumber = exports.abytes = void 0;
    exports.abool = abool;
    exports._abool2 = _abool2;
    exports._abytes2 = _abytes2;
    exports.numberToHexUnpadded = numberToHexUnpadded;
    exports.hexToNumber = hexToNumber;
    exports.bytesToNumberBE = bytesToNumberBE;
    exports.bytesToNumberLE = bytesToNumberLE;
    exports.numberToBytesBE = numberToBytesBE;
    exports.numberToBytesLE = numberToBytesLE;
    exports.numberToVarBytesBE = numberToVarBytesBE;
    exports.ensureBytes = ensureBytes;
    exports.equalBytes = equalBytes;
    exports.copyBytes = copyBytes;
    exports.asciiToBytes = asciiToBytes;
    exports.inRange = inRange;
    exports.aInRange = aInRange;
    exports.bitLen = bitLen;
    exports.bitGet = bitGet;
    exports.bitSet = bitSet;
    exports.createHmacDrbg = createHmacDrbg;
    exports.validateObject = validateObject;
    exports.isHash = isHash;
    exports._validateObject = _validateObject;
    exports.memoized = memoized;
    var utils_js_1 = require_utils();
    var utils_js_2 = require_utils();
    Object.defineProperty(exports, "abytes", { enumerable: true, get: function() {
      return utils_js_2.abytes;
    } });
    Object.defineProperty(exports, "anumber", { enumerable: true, get: function() {
      return utils_js_2.anumber;
    } });
    Object.defineProperty(exports, "bytesToHex", { enumerable: true, get: function() {
      return utils_js_2.bytesToHex;
    } });
    Object.defineProperty(exports, "bytesToUtf8", { enumerable: true, get: function() {
      return utils_js_2.bytesToUtf8;
    } });
    Object.defineProperty(exports, "concatBytes", { enumerable: true, get: function() {
      return utils_js_2.concatBytes;
    } });
    Object.defineProperty(exports, "hexToBytes", { enumerable: true, get: function() {
      return utils_js_2.hexToBytes;
    } });
    Object.defineProperty(exports, "isBytes", { enumerable: true, get: function() {
      return utils_js_2.isBytes;
    } });
    Object.defineProperty(exports, "randomBytes", { enumerable: true, get: function() {
      return utils_js_2.randomBytes;
    } });
    Object.defineProperty(exports, "utf8ToBytes", { enumerable: true, get: function() {
      return utils_js_2.utf8ToBytes;
    } });
    var _0n = /* @__PURE__ */ BigInt(0);
    var _1n = /* @__PURE__ */ BigInt(1);
    function abool(title, value) {
      if (typeof value !== "boolean")
        throw new Error(title + " boolean expected, got " + value);
    }
    function _abool2(value, title = "") {
      if (typeof value !== "boolean") {
        const prefix = title && `"${title}"`;
        throw new Error(prefix + "expected boolean, got type=" + typeof value);
      }
      return value;
    }
    function _abytes2(value, length, title = "") {
      const bytes = (0, utils_js_1.isBytes)(value);
      const len = value?.length;
      const needsLen = length !== void 0;
      if (!bytes || needsLen && len !== length) {
        const prefix = title && `"${title}" `;
        const ofLen = needsLen ? ` of length ${length}` : "";
        const got = bytes ? `length=${len}` : `type=${typeof value}`;
        throw new Error(prefix + "expected Uint8Array" + ofLen + ", got " + got);
      }
      return value;
    }
    function numberToHexUnpadded(num) {
      const hex = num.toString(16);
      return hex.length & 1 ? "0" + hex : hex;
    }
    function hexToNumber(hex) {
      if (typeof hex !== "string")
        throw new Error("hex string expected, got " + typeof hex);
      return hex === "" ? _0n : BigInt("0x" + hex);
    }
    function bytesToNumberBE(bytes) {
      return hexToNumber((0, utils_js_1.bytesToHex)(bytes));
    }
    function bytesToNumberLE(bytes) {
      (0, utils_js_1.abytes)(bytes);
      return hexToNumber((0, utils_js_1.bytesToHex)(Uint8Array.from(bytes).reverse()));
    }
    function numberToBytesBE(n, len) {
      return (0, utils_js_1.hexToBytes)(n.toString(16).padStart(len * 2, "0"));
    }
    function numberToBytesLE(n, len) {
      return numberToBytesBE(n, len).reverse();
    }
    function numberToVarBytesBE(n) {
      return (0, utils_js_1.hexToBytes)(numberToHexUnpadded(n));
    }
    function ensureBytes(title, hex, expectedLength) {
      let res;
      if (typeof hex === "string") {
        try {
          res = (0, utils_js_1.hexToBytes)(hex);
        } catch (e) {
          throw new Error(title + " must be hex string or Uint8Array, cause: " + e);
        }
      } else if ((0, utils_js_1.isBytes)(hex)) {
        res = Uint8Array.from(hex);
      } else {
        throw new Error(title + " must be hex string or Uint8Array");
      }
      const len = res.length;
      if (typeof expectedLength === "number" && len !== expectedLength)
        throw new Error(title + " of length " + expectedLength + " expected, got " + len);
      return res;
    }
    function equalBytes(a, b) {
      if (a.length !== b.length)
        return false;
      let diff = 0;
      for (let i = 0; i < a.length; i++)
        diff |= a[i] ^ b[i];
      return diff === 0;
    }
    function copyBytes(bytes) {
      return Uint8Array.from(bytes);
    }
    function asciiToBytes(ascii) {
      return Uint8Array.from(ascii, (c, i) => {
        const charCode = c.charCodeAt(0);
        if (c.length !== 1 || charCode > 127) {
          throw new Error(`string contains non-ASCII character "${ascii[i]}" with code ${charCode} at position ${i}`);
        }
        return charCode;
      });
    }
    var isPosBig = (n) => typeof n === "bigint" && _0n <= n;
    function inRange(n, min, max) {
      return isPosBig(n) && isPosBig(min) && isPosBig(max) && min <= n && n < max;
    }
    function aInRange(title, n, min, max) {
      if (!inRange(n, min, max))
        throw new Error("expected valid " + title + ": " + min + " <= n < " + max + ", got " + n);
    }
    function bitLen(n) {
      let len;
      for (len = 0; n > _0n; n >>= _1n, len += 1)
        ;
      return len;
    }
    function bitGet(n, pos) {
      return n >> BigInt(pos) & _1n;
    }
    function bitSet(n, pos, value) {
      return n | (value ? _1n : _0n) << BigInt(pos);
    }
    var bitMask = (n) => (_1n << BigInt(n)) - _1n;
    exports.bitMask = bitMask;
    function createHmacDrbg(hashLen, qByteLen, hmacFn) {
      if (typeof hashLen !== "number" || hashLen < 2)
        throw new Error("hashLen must be a number");
      if (typeof qByteLen !== "number" || qByteLen < 2)
        throw new Error("qByteLen must be a number");
      if (typeof hmacFn !== "function")
        throw new Error("hmacFn must be a function");
      const u8n = (len) => new Uint8Array(len);
      const u8of = (byte) => Uint8Array.of(byte);
      let v = u8n(hashLen);
      let k = u8n(hashLen);
      let i = 0;
      const reset = () => {
        v.fill(1);
        k.fill(0);
        i = 0;
      };
      const h = (...b) => hmacFn(k, v, ...b);
      const reseed = (seed = u8n(0)) => {
        k = h(u8of(0), seed);
        v = h();
        if (seed.length === 0)
          return;
        k = h(u8of(1), seed);
        v = h();
      };
      const gen = () => {
        if (i++ >= 1e3)
          throw new Error("drbg: tried 1000 values");
        let len = 0;
        const out = [];
        while (len < qByteLen) {
          v = h();
          const sl = v.slice();
          out.push(sl);
          len += v.length;
        }
        return (0, utils_js_1.concatBytes)(...out);
      };
      const genUntil = (seed, pred) => {
        reset();
        reseed(seed);
        let res = void 0;
        while (!(res = pred(gen())))
          reseed();
        reset();
        return res;
      };
      return genUntil;
    }
    var validatorFns = {
      bigint: (val) => typeof val === "bigint",
      function: (val) => typeof val === "function",
      boolean: (val) => typeof val === "boolean",
      string: (val) => typeof val === "string",
      stringOrUint8Array: (val) => typeof val === "string" || (0, utils_js_1.isBytes)(val),
      isSafeInteger: (val) => Number.isSafeInteger(val),
      array: (val) => Array.isArray(val),
      field: (val, object) => object.Fp.isValid(val),
      hash: (val) => typeof val === "function" && Number.isSafeInteger(val.outputLen)
    };
    function validateObject(object, validators, optValidators = {}) {
      const checkField = (fieldName, type, isOptional) => {
        const checkVal = validatorFns[type];
        if (typeof checkVal !== "function")
          throw new Error("invalid validator function");
        const val = object[fieldName];
        if (isOptional && val === void 0)
          return;
        if (!checkVal(val, object)) {
          throw new Error("param " + String(fieldName) + " is invalid. Expected " + type + ", got " + val);
        }
      };
      for (const [fieldName, type] of Object.entries(validators))
        checkField(fieldName, type, false);
      for (const [fieldName, type] of Object.entries(optValidators))
        checkField(fieldName, type, true);
      return object;
    }
    function isHash(val) {
      return typeof val === "function" && Number.isSafeInteger(val.outputLen);
    }
    function _validateObject(object, fields, optFields = {}) {
      if (!object || typeof object !== "object")
        throw new Error("expected valid options object");
      function checkField(fieldName, expectedType, isOpt) {
        const val = object[fieldName];
        if (isOpt && val === void 0)
          return;
        const current = typeof val;
        if (current !== expectedType || val === null)
          throw new Error(`param "${fieldName}" is invalid: expected ${expectedType}, got ${current}`);
      }
      Object.entries(fields).forEach(([k, v]) => checkField(k, v, false));
      Object.entries(optFields).forEach(([k, v]) => checkField(k, v, true));
    }
    var notImplemented = () => {
      throw new Error("not implemented");
    };
    exports.notImplemented = notImplemented;
    function memoized(fn) {
      const map = /* @__PURE__ */ new WeakMap();
      return (arg, ...args) => {
        const val = map.get(arg);
        if (val !== void 0)
          return val;
        const computed = fn(arg, ...args);
        map.set(arg, computed);
        return computed;
      };
    }
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/modular.js
var require_modular = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/modular.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.isNegativeLE = void 0;
    exports.mod = mod;
    exports.pow = pow;
    exports.pow2 = pow2;
    exports.invert = invert;
    exports.tonelliShanks = tonelliShanks;
    exports.FpSqrt = FpSqrt;
    exports.validateField = validateField;
    exports.FpPow = FpPow;
    exports.FpInvertBatch = FpInvertBatch;
    exports.FpDiv = FpDiv;
    exports.FpLegendre = FpLegendre;
    exports.FpIsSquare = FpIsSquare;
    exports.nLength = nLength;
    exports.Field = Field;
    exports.FpSqrtOdd = FpSqrtOdd;
    exports.FpSqrtEven = FpSqrtEven;
    exports.hashToPrivateScalar = hashToPrivateScalar;
    exports.getFieldBytesLength = getFieldBytesLength;
    exports.getMinHashLength = getMinHashLength;
    exports.mapHashToField = mapHashToField;
    var utils_ts_1 = require_utils2();
    var _0n = BigInt(0);
    var _1n = BigInt(1);
    var _2n = /* @__PURE__ */ BigInt(2);
    var _3n = /* @__PURE__ */ BigInt(3);
    var _4n = /* @__PURE__ */ BigInt(4);
    var _5n = /* @__PURE__ */ BigInt(5);
    var _7n = /* @__PURE__ */ BigInt(7);
    var _8n = /* @__PURE__ */ BigInt(8);
    var _9n = /* @__PURE__ */ BigInt(9);
    var _16n = /* @__PURE__ */ BigInt(16);
    function mod(a, b) {
      const result = a % b;
      return result >= _0n ? result : b + result;
    }
    function pow(num, power, modulo) {
      return FpPow(Field(modulo), num, power);
    }
    function pow2(x, power, modulo) {
      let res = x;
      while (power-- > _0n) {
        res *= res;
        res %= modulo;
      }
      return res;
    }
    function invert(number, modulo) {
      if (number === _0n)
        throw new Error("invert: expected non-zero number");
      if (modulo <= _0n)
        throw new Error("invert: expected positive modulus, got " + modulo);
      let a = mod(number, modulo);
      let b = modulo;
      let x = _0n, y = _1n, u = _1n, v = _0n;
      while (a !== _0n) {
        const q = b / a;
        const r = b % a;
        const m = x - u * q;
        const n = y - v * q;
        b = a, a = r, x = u, y = v, u = m, v = n;
      }
      const gcd = b;
      if (gcd !== _1n)
        throw new Error("invert: does not exist");
      return mod(x, modulo);
    }
    function assertIsSquare(Fp, root, n) {
      if (!Fp.eql(Fp.sqr(root), n))
        throw new Error("Cannot find square root");
    }
    function sqrt3mod4(Fp, n) {
      const p1div4 = (Fp.ORDER + _1n) / _4n;
      const root = Fp.pow(n, p1div4);
      assertIsSquare(Fp, root, n);
      return root;
    }
    function sqrt5mod8(Fp, n) {
      const p5div8 = (Fp.ORDER - _5n) / _8n;
      const n2 = Fp.mul(n, _2n);
      const v = Fp.pow(n2, p5div8);
      const nv = Fp.mul(n, v);
      const i = Fp.mul(Fp.mul(nv, _2n), v);
      const root = Fp.mul(nv, Fp.sub(i, Fp.ONE));
      assertIsSquare(Fp, root, n);
      return root;
    }
    function sqrt9mod16(P) {
      const Fp_ = Field(P);
      const tn = tonelliShanks(P);
      const c1 = tn(Fp_, Fp_.neg(Fp_.ONE));
      const c2 = tn(Fp_, c1);
      const c3 = tn(Fp_, Fp_.neg(c1));
      const c4 = (P + _7n) / _16n;
      return (Fp, n) => {
        let tv1 = Fp.pow(n, c4);
        let tv2 = Fp.mul(tv1, c1);
        const tv3 = Fp.mul(tv1, c2);
        const tv4 = Fp.mul(tv1, c3);
        const e1 = Fp.eql(Fp.sqr(tv2), n);
        const e2 = Fp.eql(Fp.sqr(tv3), n);
        tv1 = Fp.cmov(tv1, tv2, e1);
        tv2 = Fp.cmov(tv4, tv3, e2);
        const e3 = Fp.eql(Fp.sqr(tv2), n);
        const root = Fp.cmov(tv1, tv2, e3);
        assertIsSquare(Fp, root, n);
        return root;
      };
    }
    function tonelliShanks(P) {
      if (P < _3n)
        throw new Error("sqrt is not defined for small field");
      let Q = P - _1n;
      let S = 0;
      while (Q % _2n === _0n) {
        Q /= _2n;
        S++;
      }
      let Z = _2n;
      const _Fp = Field(P);
      while (FpLegendre(_Fp, Z) === 1) {
        if (Z++ > 1e3)
          throw new Error("Cannot find square root: probably non-prime P");
      }
      if (S === 1)
        return sqrt3mod4;
      let cc = _Fp.pow(Z, Q);
      const Q1div2 = (Q + _1n) / _2n;
      return function tonelliSlow(Fp, n) {
        if (Fp.is0(n))
          return n;
        if (FpLegendre(Fp, n) !== 1)
          throw new Error("Cannot find square root");
        let M = S;
        let c = Fp.mul(Fp.ONE, cc);
        let t = Fp.pow(n, Q);
        let R = Fp.pow(n, Q1div2);
        while (!Fp.eql(t, Fp.ONE)) {
          if (Fp.is0(t))
            return Fp.ZERO;
          let i = 1;
          let t_tmp = Fp.sqr(t);
          while (!Fp.eql(t_tmp, Fp.ONE)) {
            i++;
            t_tmp = Fp.sqr(t_tmp);
            if (i === M)
              throw new Error("Cannot find square root");
          }
          const exponent = _1n << BigInt(M - i - 1);
          const b = Fp.pow(c, exponent);
          M = i;
          c = Fp.sqr(b);
          t = Fp.mul(t, c);
          R = Fp.mul(R, b);
        }
        return R;
      };
    }
    function FpSqrt(P) {
      if (P % _4n === _3n)
        return sqrt3mod4;
      if (P % _8n === _5n)
        return sqrt5mod8;
      if (P % _16n === _9n)
        return sqrt9mod16(P);
      return tonelliShanks(P);
    }
    var isNegativeLE = (num, modulo) => (mod(num, modulo) & _1n) === _1n;
    exports.isNegativeLE = isNegativeLE;
    var FIELD_FIELDS = [
      "create",
      "isValid",
      "is0",
      "neg",
      "inv",
      "sqrt",
      "sqr",
      "eql",
      "add",
      "sub",
      "mul",
      "pow",
      "div",
      "addN",
      "subN",
      "mulN",
      "sqrN"
    ];
    function validateField(field) {
      const initial = {
        ORDER: "bigint",
        MASK: "bigint",
        BYTES: "number",
        BITS: "number"
      };
      const opts = FIELD_FIELDS.reduce((map, val) => {
        map[val] = "function";
        return map;
      }, initial);
      (0, utils_ts_1._validateObject)(field, opts);
      return field;
    }
    function FpPow(Fp, num, power) {
      if (power < _0n)
        throw new Error("invalid exponent, negatives unsupported");
      if (power === _0n)
        return Fp.ONE;
      if (power === _1n)
        return num;
      let p = Fp.ONE;
      let d = num;
      while (power > _0n) {
        if (power & _1n)
          p = Fp.mul(p, d);
        d = Fp.sqr(d);
        power >>= _1n;
      }
      return p;
    }
    function FpInvertBatch(Fp, nums, passZero = false) {
      const inverted = new Array(nums.length).fill(passZero ? Fp.ZERO : void 0);
      const multipliedAcc = nums.reduce((acc, num, i) => {
        if (Fp.is0(num))
          return acc;
        inverted[i] = acc;
        return Fp.mul(acc, num);
      }, Fp.ONE);
      const invertedAcc = Fp.inv(multipliedAcc);
      nums.reduceRight((acc, num, i) => {
        if (Fp.is0(num))
          return acc;
        inverted[i] = Fp.mul(acc, inverted[i]);
        return Fp.mul(acc, num);
      }, invertedAcc);
      return inverted;
    }
    function FpDiv(Fp, lhs, rhs) {
      return Fp.mul(lhs, typeof rhs === "bigint" ? invert(rhs, Fp.ORDER) : Fp.inv(rhs));
    }
    function FpLegendre(Fp, n) {
      const p1mod2 = (Fp.ORDER - _1n) / _2n;
      const powered = Fp.pow(n, p1mod2);
      const yes = Fp.eql(powered, Fp.ONE);
      const zero = Fp.eql(powered, Fp.ZERO);
      const no = Fp.eql(powered, Fp.neg(Fp.ONE));
      if (!yes && !zero && !no)
        throw new Error("invalid Legendre symbol result");
      return yes ? 1 : zero ? 0 : -1;
    }
    function FpIsSquare(Fp, n) {
      const l = FpLegendre(Fp, n);
      return l === 1;
    }
    function nLength(n, nBitLength) {
      if (nBitLength !== void 0)
        (0, utils_ts_1.anumber)(nBitLength);
      const _nBitLength = nBitLength !== void 0 ? nBitLength : n.toString(2).length;
      const nByteLength = Math.ceil(_nBitLength / 8);
      return { nBitLength: _nBitLength, nByteLength };
    }
    function Field(ORDER, bitLenOrOpts, isLE = false, opts = {}) {
      if (ORDER <= _0n)
        throw new Error("invalid field: expected ORDER > 0, got " + ORDER);
      let _nbitLength = void 0;
      let _sqrt = void 0;
      let modFromBytes = false;
      let allowedLengths = void 0;
      if (typeof bitLenOrOpts === "object" && bitLenOrOpts != null) {
        if (opts.sqrt || isLE)
          throw new Error("cannot specify opts in two arguments");
        const _opts = bitLenOrOpts;
        if (_opts.BITS)
          _nbitLength = _opts.BITS;
        if (_opts.sqrt)
          _sqrt = _opts.sqrt;
        if (typeof _opts.isLE === "boolean")
          isLE = _opts.isLE;
        if (typeof _opts.modFromBytes === "boolean")
          modFromBytes = _opts.modFromBytes;
        allowedLengths = _opts.allowedLengths;
      } else {
        if (typeof bitLenOrOpts === "number")
          _nbitLength = bitLenOrOpts;
        if (opts.sqrt)
          _sqrt = opts.sqrt;
      }
      const { nBitLength: BITS, nByteLength: BYTES } = nLength(ORDER, _nbitLength);
      if (BYTES > 2048)
        throw new Error("invalid field: expected ORDER of <= 2048 bytes");
      let sqrtP;
      const f = Object.freeze({
        ORDER,
        isLE,
        BITS,
        BYTES,
        MASK: (0, utils_ts_1.bitMask)(BITS),
        ZERO: _0n,
        ONE: _1n,
        allowedLengths,
        create: (num) => mod(num, ORDER),
        isValid: (num) => {
          if (typeof num !== "bigint")
            throw new Error("invalid field element: expected bigint, got " + typeof num);
          return _0n <= num && num < ORDER;
        },
        is0: (num) => num === _0n,
        // is valid and invertible
        isValidNot0: (num) => !f.is0(num) && f.isValid(num),
        isOdd: (num) => (num & _1n) === _1n,
        neg: (num) => mod(-num, ORDER),
        eql: (lhs, rhs) => lhs === rhs,
        sqr: (num) => mod(num * num, ORDER),
        add: (lhs, rhs) => mod(lhs + rhs, ORDER),
        sub: (lhs, rhs) => mod(lhs - rhs, ORDER),
        mul: (lhs, rhs) => mod(lhs * rhs, ORDER),
        pow: (num, power) => FpPow(f, num, power),
        div: (lhs, rhs) => mod(lhs * invert(rhs, ORDER), ORDER),
        // Same as above, but doesn't normalize
        sqrN: (num) => num * num,
        addN: (lhs, rhs) => lhs + rhs,
        subN: (lhs, rhs) => lhs - rhs,
        mulN: (lhs, rhs) => lhs * rhs,
        inv: (num) => invert(num, ORDER),
        sqrt: _sqrt || ((n) => {
          if (!sqrtP)
            sqrtP = FpSqrt(ORDER);
          return sqrtP(f, n);
        }),
        toBytes: (num) => isLE ? (0, utils_ts_1.numberToBytesLE)(num, BYTES) : (0, utils_ts_1.numberToBytesBE)(num, BYTES),
        fromBytes: (bytes, skipValidation = true) => {
          if (allowedLengths) {
            if (!allowedLengths.includes(bytes.length) || bytes.length > BYTES) {
              throw new Error("Field.fromBytes: expected " + allowedLengths + " bytes, got " + bytes.length);
            }
            const padded = new Uint8Array(BYTES);
            padded.set(bytes, isLE ? 0 : padded.length - bytes.length);
            bytes = padded;
          }
          if (bytes.length !== BYTES)
            throw new Error("Field.fromBytes: expected " + BYTES + " bytes, got " + bytes.length);
          let scalar = isLE ? (0, utils_ts_1.bytesToNumberLE)(bytes) : (0, utils_ts_1.bytesToNumberBE)(bytes);
          if (modFromBytes)
            scalar = mod(scalar, ORDER);
          if (!skipValidation) {
            if (!f.isValid(scalar))
              throw new Error("invalid field element: outside of range 0..ORDER");
          }
          return scalar;
        },
        // TODO: we don't need it here, move out to separate fn
        invertBatch: (lst) => FpInvertBatch(f, lst),
        // We can't move this out because Fp6, Fp12 implement it
        // and it's unclear what to return in there.
        cmov: (a, b, c) => c ? b : a
      });
      return Object.freeze(f);
    }
    function FpSqrtOdd(Fp, elm) {
      if (!Fp.isOdd)
        throw new Error("Field doesn't have isOdd");
      const root = Fp.sqrt(elm);
      return Fp.isOdd(root) ? root : Fp.neg(root);
    }
    function FpSqrtEven(Fp, elm) {
      if (!Fp.isOdd)
        throw new Error("Field doesn't have isOdd");
      const root = Fp.sqrt(elm);
      return Fp.isOdd(root) ? Fp.neg(root) : root;
    }
    function hashToPrivateScalar(hash, groupOrder, isLE = false) {
      hash = (0, utils_ts_1.ensureBytes)("privateHash", hash);
      const hashLen = hash.length;
      const minLen = nLength(groupOrder).nByteLength + 8;
      if (minLen < 24 || hashLen < minLen || hashLen > 1024)
        throw new Error("hashToPrivateScalar: expected " + minLen + "-1024 bytes of input, got " + hashLen);
      const num = isLE ? (0, utils_ts_1.bytesToNumberLE)(hash) : (0, utils_ts_1.bytesToNumberBE)(hash);
      return mod(num, groupOrder - _1n) + _1n;
    }
    function getFieldBytesLength(fieldOrder) {
      if (typeof fieldOrder !== "bigint")
        throw new Error("field order must be bigint");
      const bitLength = fieldOrder.toString(2).length;
      return Math.ceil(bitLength / 8);
    }
    function getMinHashLength(fieldOrder) {
      const length = getFieldBytesLength(fieldOrder);
      return length + Math.ceil(length / 2);
    }
    function mapHashToField(key, fieldOrder, isLE = false) {
      const len = key.length;
      const fieldLen = getFieldBytesLength(fieldOrder);
      const minLen = getMinHashLength(fieldOrder);
      if (len < 16 || len < minLen || len > 1024)
        throw new Error("expected " + minLen + "-1024 bytes of input, got " + len);
      const num = isLE ? (0, utils_ts_1.bytesToNumberLE)(key) : (0, utils_ts_1.bytesToNumberBE)(key);
      const reduced = mod(num, fieldOrder - _1n) + _1n;
      return isLE ? (0, utils_ts_1.numberToBytesLE)(reduced, fieldLen) : (0, utils_ts_1.numberToBytesBE)(reduced, fieldLen);
    }
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/curve.js
var require_curve = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/curve.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.wNAF = void 0;
    exports.negateCt = negateCt;
    exports.normalizeZ = normalizeZ;
    exports.mulEndoUnsafe = mulEndoUnsafe;
    exports.pippenger = pippenger;
    exports.precomputeMSMUnsafe = precomputeMSMUnsafe;
    exports.validateBasic = validateBasic;
    exports._createCurveFields = _createCurveFields;
    var utils_ts_1 = require_utils2();
    var modular_ts_1 = require_modular();
    var _0n = BigInt(0);
    var _1n = BigInt(1);
    function negateCt(condition, item) {
      const neg = item.negate();
      return condition ? neg : item;
    }
    function normalizeZ(c, points) {
      const invertedZs = (0, modular_ts_1.FpInvertBatch)(c.Fp, points.map((p) => p.Z));
      return points.map((p, i) => c.fromAffine(p.toAffine(invertedZs[i])));
    }
    function validateW(W, bits) {
      if (!Number.isSafeInteger(W) || W <= 0 || W > bits)
        throw new Error("invalid window size, expected [1.." + bits + "], got W=" + W);
    }
    function calcWOpts(W, scalarBits) {
      validateW(W, scalarBits);
      const windows = Math.ceil(scalarBits / W) + 1;
      const windowSize = 2 ** (W - 1);
      const maxNumber = 2 ** W;
      const mask = (0, utils_ts_1.bitMask)(W);
      const shiftBy = BigInt(W);
      return { windows, windowSize, mask, maxNumber, shiftBy };
    }
    function calcOffsets(n, window, wOpts) {
      const { windowSize, mask, maxNumber, shiftBy } = wOpts;
      let wbits = Number(n & mask);
      let nextN = n >> shiftBy;
      if (wbits > windowSize) {
        wbits -= maxNumber;
        nextN += _1n;
      }
      const offsetStart = window * windowSize;
      const offset = offsetStart + Math.abs(wbits) - 1;
      const isZero = wbits === 0;
      const isNeg = wbits < 0;
      const isNegF = window % 2 !== 0;
      const offsetF = offsetStart;
      return { nextN, offset, isZero, isNeg, isNegF, offsetF };
    }
    function validateMSMPoints(points, c) {
      if (!Array.isArray(points))
        throw new Error("array expected");
      points.forEach((p, i) => {
        if (!(p instanceof c))
          throw new Error("invalid point at index " + i);
      });
    }
    function validateMSMScalars(scalars, field) {
      if (!Array.isArray(scalars))
        throw new Error("array of scalars expected");
      scalars.forEach((s, i) => {
        if (!field.isValid(s))
          throw new Error("invalid scalar at index " + i);
      });
    }
    var pointPrecomputes = /* @__PURE__ */ new WeakMap();
    var pointWindowSizes = /* @__PURE__ */ new WeakMap();
    function getW(P) {
      return pointWindowSizes.get(P) || 1;
    }
    function assert0(n) {
      if (n !== _0n)
        throw new Error("invalid wNAF");
    }
    var wNAF = class {
      // Parametrized with a given Point class (not individual point)
      constructor(Point, bits) {
        this.BASE = Point.BASE;
        this.ZERO = Point.ZERO;
        this.Fn = Point.Fn;
        this.bits = bits;
      }
      // non-const time multiplication ladder
      _unsafeLadder(elm, n, p = this.ZERO) {
        let d = elm;
        while (n > _0n) {
          if (n & _1n)
            p = p.add(d);
          d = d.double();
          n >>= _1n;
        }
        return p;
      }
      /**
       * Creates a wNAF precomputation window. Used for caching.
       * Default window size is set by `utils.precompute()` and is equal to 8.
       * Number of precomputed points depends on the curve size:
       * 2^(𝑊−1) * (Math.ceil(𝑛 / 𝑊) + 1), where:
       * - 𝑊 is the window size
       * - 𝑛 is the bitlength of the curve order.
       * For a 256-bit curve and window size 8, the number of precomputed points is 128 * 33 = 4224.
       * @param point Point instance
       * @param W window size
       * @returns precomputed point tables flattened to a single array
       */
      precomputeWindow(point, W) {
        const { windows, windowSize } = calcWOpts(W, this.bits);
        const points = [];
        let p = point;
        let base = p;
        for (let window = 0; window < windows; window++) {
          base = p;
          points.push(base);
          for (let i = 1; i < windowSize; i++) {
            base = base.add(p);
            points.push(base);
          }
          p = base.double();
        }
        return points;
      }
      /**
       * Implements ec multiplication using precomputed tables and w-ary non-adjacent form.
       * More compact implementation:
       * https://github.com/paulmillr/noble-secp256k1/blob/47cb1669b6e506ad66b35fe7d76132ae97465da2/index.ts#L502-L541
       * @returns real and fake (for const-time) points
       */
      wNAF(W, precomputes, n) {
        if (!this.Fn.isValid(n))
          throw new Error("invalid scalar");
        let p = this.ZERO;
        let f = this.BASE;
        const wo = calcWOpts(W, this.bits);
        for (let window = 0; window < wo.windows; window++) {
          const { nextN, offset, isZero, isNeg, isNegF, offsetF } = calcOffsets(n, window, wo);
          n = nextN;
          if (isZero) {
            f = f.add(negateCt(isNegF, precomputes[offsetF]));
          } else {
            p = p.add(negateCt(isNeg, precomputes[offset]));
          }
        }
        assert0(n);
        return { p, f };
      }
      /**
       * Implements ec unsafe (non const-time) multiplication using precomputed tables and w-ary non-adjacent form.
       * @param acc accumulator point to add result of multiplication
       * @returns point
       */
      wNAFUnsafe(W, precomputes, n, acc = this.ZERO) {
        const wo = calcWOpts(W, this.bits);
        for (let window = 0; window < wo.windows; window++) {
          if (n === _0n)
            break;
          const { nextN, offset, isZero, isNeg } = calcOffsets(n, window, wo);
          n = nextN;
          if (isZero) {
            continue;
          } else {
            const item = precomputes[offset];
            acc = acc.add(isNeg ? item.negate() : item);
          }
        }
        assert0(n);
        return acc;
      }
      getPrecomputes(W, point, transform) {
        let comp = pointPrecomputes.get(point);
        if (!comp) {
          comp = this.precomputeWindow(point, W);
          if (W !== 1) {
            if (typeof transform === "function")
              comp = transform(comp);
            pointPrecomputes.set(point, comp);
          }
        }
        return comp;
      }
      cached(point, scalar, transform) {
        const W = getW(point);
        return this.wNAF(W, this.getPrecomputes(W, point, transform), scalar);
      }
      unsafe(point, scalar, transform, prev) {
        const W = getW(point);
        if (W === 1)
          return this._unsafeLadder(point, scalar, prev);
        return this.wNAFUnsafe(W, this.getPrecomputes(W, point, transform), scalar, prev);
      }
      // We calculate precomputes for elliptic curve point multiplication
      // using windowed method. This specifies window size and
      // stores precomputed values. Usually only base point would be precomputed.
      createCache(P, W) {
        validateW(W, this.bits);
        pointWindowSizes.set(P, W);
        pointPrecomputes.delete(P);
      }
      hasCache(elm) {
        return getW(elm) !== 1;
      }
    };
    exports.wNAF = wNAF;
    function mulEndoUnsafe(Point, point, k1, k2) {
      let acc = point;
      let p1 = Point.ZERO;
      let p2 = Point.ZERO;
      while (k1 > _0n || k2 > _0n) {
        if (k1 & _1n)
          p1 = p1.add(acc);
        if (k2 & _1n)
          p2 = p2.add(acc);
        acc = acc.double();
        k1 >>= _1n;
        k2 >>= _1n;
      }
      return { p1, p2 };
    }
    function pippenger(c, fieldN, points, scalars) {
      validateMSMPoints(points, c);
      validateMSMScalars(scalars, fieldN);
      const plength = points.length;
      const slength = scalars.length;
      if (plength !== slength)
        throw new Error("arrays of points and scalars must have equal length");
      const zero = c.ZERO;
      const wbits = (0, utils_ts_1.bitLen)(BigInt(plength));
      let windowSize = 1;
      if (wbits > 12)
        windowSize = wbits - 3;
      else if (wbits > 4)
        windowSize = wbits - 2;
      else if (wbits > 0)
        windowSize = 2;
      const MASK = (0, utils_ts_1.bitMask)(windowSize);
      const buckets = new Array(Number(MASK) + 1).fill(zero);
      const lastBits = Math.floor((fieldN.BITS - 1) / windowSize) * windowSize;
      let sum = zero;
      for (let i = lastBits; i >= 0; i -= windowSize) {
        buckets.fill(zero);
        for (let j = 0; j < slength; j++) {
          const scalar = scalars[j];
          const wbits2 = Number(scalar >> BigInt(i) & MASK);
          buckets[wbits2] = buckets[wbits2].add(points[j]);
        }
        let resI = zero;
        for (let j = buckets.length - 1, sumI = zero; j > 0; j--) {
          sumI = sumI.add(buckets[j]);
          resI = resI.add(sumI);
        }
        sum = sum.add(resI);
        if (i !== 0)
          for (let j = 0; j < windowSize; j++)
            sum = sum.double();
      }
      return sum;
    }
    function precomputeMSMUnsafe(c, fieldN, points, windowSize) {
      validateW(windowSize, fieldN.BITS);
      validateMSMPoints(points, c);
      const zero = c.ZERO;
      const tableSize = 2 ** windowSize - 1;
      const chunks = Math.ceil(fieldN.BITS / windowSize);
      const MASK = (0, utils_ts_1.bitMask)(windowSize);
      const tables = points.map((p) => {
        const res = [];
        for (let i = 0, acc = p; i < tableSize; i++) {
          res.push(acc);
          acc = acc.add(p);
        }
        return res;
      });
      return (scalars) => {
        validateMSMScalars(scalars, fieldN);
        if (scalars.length > points.length)
          throw new Error("array of scalars must be smaller than array of points");
        let res = zero;
        for (let i = 0; i < chunks; i++) {
          if (res !== zero)
            for (let j = 0; j < windowSize; j++)
              res = res.double();
          const shiftBy = BigInt(chunks * windowSize - (i + 1) * windowSize);
          for (let j = 0; j < scalars.length; j++) {
            const n = scalars[j];
            const curr = Number(n >> shiftBy & MASK);
            if (!curr)
              continue;
            res = res.add(tables[j][curr - 1]);
          }
        }
        return res;
      };
    }
    function validateBasic(curve) {
      (0, modular_ts_1.validateField)(curve.Fp);
      (0, utils_ts_1.validateObject)(curve, {
        n: "bigint",
        h: "bigint",
        Gx: "field",
        Gy: "field"
      }, {
        nBitLength: "isSafeInteger",
        nByteLength: "isSafeInteger"
      });
      return Object.freeze({
        ...(0, modular_ts_1.nLength)(curve.n, curve.nBitLength),
        ...curve,
        ...{ p: curve.Fp.ORDER }
      });
    }
    function createField(order, field, isLE) {
      if (field) {
        if (field.ORDER !== order)
          throw new Error("Field.ORDER must match order: Fp == p, Fn == n");
        (0, modular_ts_1.validateField)(field);
        return field;
      } else {
        return (0, modular_ts_1.Field)(order, { isLE });
      }
    }
    function _createCurveFields(type, CURVE, curveOpts = {}, FpFnLE) {
      if (FpFnLE === void 0)
        FpFnLE = type === "edwards";
      if (!CURVE || typeof CURVE !== "object")
        throw new Error(`expected valid ${type} CURVE object`);
      for (const p of ["p", "n", "h"]) {
        const val = CURVE[p];
        if (!(typeof val === "bigint" && val > _0n))
          throw new Error(`CURVE.${p} must be positive bigint`);
      }
      const Fp = createField(CURVE.p, curveOpts.Fp, FpFnLE);
      const Fn = createField(CURVE.n, curveOpts.Fn, FpFnLE);
      const _b = type === "weierstrass" ? "b" : "d";
      const params = ["Gx", "Gy", "a", _b];
      for (const p of params) {
        if (!Fp.isValid(CURVE[p]))
          throw new Error(`CURVE.${p} must be valid field element of CURVE.Fp`);
      }
      CURVE = Object.freeze(Object.assign({}, CURVE));
      return { CURVE, Fp, Fn };
    }
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/hash-to-curve.js
var require_hash_to_curve = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/hash-to-curve.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports._DST_scalar = void 0;
    exports.expand_message_xmd = expand_message_xmd;
    exports.expand_message_xof = expand_message_xof;
    exports.hash_to_field = hash_to_field;
    exports.isogenyMap = isogenyMap;
    exports.createHasher = createHasher;
    var utils_ts_1 = require_utils2();
    var modular_ts_1 = require_modular();
    var os2ip = utils_ts_1.bytesToNumberBE;
    function i2osp(value, length) {
      anum(value);
      anum(length);
      if (value < 0 || value >= 1 << 8 * length)
        throw new Error("invalid I2OSP input: " + value);
      const res = Array.from({ length }).fill(0);
      for (let i = length - 1; i >= 0; i--) {
        res[i] = value & 255;
        value >>>= 8;
      }
      return new Uint8Array(res);
    }
    function strxor(a, b) {
      const arr = new Uint8Array(a.length);
      for (let i = 0; i < a.length; i++) {
        arr[i] = a[i] ^ b[i];
      }
      return arr;
    }
    function anum(item) {
      if (!Number.isSafeInteger(item))
        throw new Error("number expected");
    }
    function normDST(DST) {
      if (!(0, utils_ts_1.isBytes)(DST) && typeof DST !== "string")
        throw new Error("DST must be Uint8Array or string");
      return typeof DST === "string" ? (0, utils_ts_1.utf8ToBytes)(DST) : DST;
    }
    function expand_message_xmd(msg, DST, lenInBytes, H) {
      (0, utils_ts_1.abytes)(msg);
      anum(lenInBytes);
      DST = normDST(DST);
      if (DST.length > 255)
        DST = H((0, utils_ts_1.concatBytes)((0, utils_ts_1.utf8ToBytes)("H2C-OVERSIZE-DST-"), DST));
      const { outputLen: b_in_bytes, blockLen: r_in_bytes } = H;
      const ell = Math.ceil(lenInBytes / b_in_bytes);
      if (lenInBytes > 65535 || ell > 255)
        throw new Error("expand_message_xmd: invalid lenInBytes");
      const DST_prime = (0, utils_ts_1.concatBytes)(DST, i2osp(DST.length, 1));
      const Z_pad = i2osp(0, r_in_bytes);
      const l_i_b_str = i2osp(lenInBytes, 2);
      const b = new Array(ell);
      const b_0 = H((0, utils_ts_1.concatBytes)(Z_pad, msg, l_i_b_str, i2osp(0, 1), DST_prime));
      b[0] = H((0, utils_ts_1.concatBytes)(b_0, i2osp(1, 1), DST_prime));
      for (let i = 1; i <= ell; i++) {
        const args = [strxor(b_0, b[i - 1]), i2osp(i + 1, 1), DST_prime];
        b[i] = H((0, utils_ts_1.concatBytes)(...args));
      }
      const pseudo_random_bytes = (0, utils_ts_1.concatBytes)(...b);
      return pseudo_random_bytes.slice(0, lenInBytes);
    }
    function expand_message_xof(msg, DST, lenInBytes, k, H) {
      (0, utils_ts_1.abytes)(msg);
      anum(lenInBytes);
      DST = normDST(DST);
      if (DST.length > 255) {
        const dkLen = Math.ceil(2 * k / 8);
        DST = H.create({ dkLen }).update((0, utils_ts_1.utf8ToBytes)("H2C-OVERSIZE-DST-")).update(DST).digest();
      }
      if (lenInBytes > 65535 || DST.length > 255)
        throw new Error("expand_message_xof: invalid lenInBytes");
      return H.create({ dkLen: lenInBytes }).update(msg).update(i2osp(lenInBytes, 2)).update(DST).update(i2osp(DST.length, 1)).digest();
    }
    function hash_to_field(msg, count, options) {
      (0, utils_ts_1._validateObject)(options, {
        p: "bigint",
        m: "number",
        k: "number",
        hash: "function"
      });
      const { p, k, m, hash, expand, DST } = options;
      if (!(0, utils_ts_1.isHash)(options.hash))
        throw new Error("expected valid hash");
      (0, utils_ts_1.abytes)(msg);
      anum(count);
      const log2p = p.toString(2).length;
      const L = Math.ceil((log2p + k) / 8);
      const len_in_bytes = count * m * L;
      let prb;
      if (expand === "xmd") {
        prb = expand_message_xmd(msg, DST, len_in_bytes, hash);
      } else if (expand === "xof") {
        prb = expand_message_xof(msg, DST, len_in_bytes, k, hash);
      } else if (expand === "_internal_pass") {
        prb = msg;
      } else {
        throw new Error('expand must be "xmd" or "xof"');
      }
      const u = new Array(count);
      for (let i = 0; i < count; i++) {
        const e = new Array(m);
        for (let j = 0; j < m; j++) {
          const elm_offset = L * (j + i * m);
          const tv = prb.subarray(elm_offset, elm_offset + L);
          e[j] = (0, modular_ts_1.mod)(os2ip(tv), p);
        }
        u[i] = e;
      }
      return u;
    }
    function isogenyMap(field, map) {
      const coeff = map.map((i) => Array.from(i).reverse());
      return (x, y) => {
        const [xn, xd, yn, yd] = coeff.map((val) => val.reduce((acc, i) => field.add(field.mul(acc, x), i)));
        const [xd_inv, yd_inv] = (0, modular_ts_1.FpInvertBatch)(field, [xd, yd], true);
        x = field.mul(xn, xd_inv);
        y = field.mul(y, field.mul(yn, yd_inv));
        return { x, y };
      };
    }
    exports._DST_scalar = (0, utils_ts_1.utf8ToBytes)("HashToScalar-");
    function createHasher(Point, mapToCurve, defaults) {
      if (typeof mapToCurve !== "function")
        throw new Error("mapToCurve() must be defined");
      function map(num) {
        return Point.fromAffine(mapToCurve(num));
      }
      function clear(initial) {
        const P = initial.clearCofactor();
        if (P.equals(Point.ZERO))
          return Point.ZERO;
        P.assertValidity();
        return P;
      }
      return {
        defaults,
        hashToCurve(msg, options) {
          const opts = Object.assign({}, defaults, options);
          const u = hash_to_field(msg, 2, opts);
          const u0 = map(u[0]);
          const u1 = map(u[1]);
          return clear(u0.add(u1));
        },
        encodeToCurve(msg, options) {
          const optsDst = defaults.encodeDST ? { DST: defaults.encodeDST } : {};
          const opts = Object.assign({}, defaults, optsDst, options);
          const u = hash_to_field(msg, 1, opts);
          const u0 = map(u[0]);
          return clear(u0);
        },
        /** See {@link H2CHasher} */
        mapToCurve(scalars) {
          if (!Array.isArray(scalars))
            throw new Error("expected array of bigints");
          for (const i of scalars)
            if (typeof i !== "bigint")
              throw new Error("expected array of bigints");
          return clear(map(scalars));
        },
        // hash_to_scalar can produce 0: https://www.rfc-editor.org/errata/eid8393
        // RFC 9380, draft-irtf-cfrg-bbs-signatures-08
        hashToScalar(msg, options) {
          const N = Point.Fn.ORDER;
          const opts = Object.assign({}, defaults, { p: N, m: 1, DST: exports._DST_scalar }, options);
          return hash_to_field(msg, 1, opts)[0][0];
        }
      };
    }
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/hmac.js
var require_hmac = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/hashes/hmac.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.hmac = exports.HMAC = void 0;
    var utils_ts_1 = require_utils();
    var HMAC = class extends utils_ts_1.Hash {
      constructor(hash, _key) {
        super();
        this.finished = false;
        this.destroyed = false;
        (0, utils_ts_1.ahash)(hash);
        const key = (0, utils_ts_1.toBytes)(_key);
        this.iHash = hash.create();
        if (typeof this.iHash.update !== "function")
          throw new Error("Expected instance of class which extends utils.Hash");
        this.blockLen = this.iHash.blockLen;
        this.outputLen = this.iHash.outputLen;
        const blockLen = this.blockLen;
        const pad = new Uint8Array(blockLen);
        pad.set(key.length > blockLen ? hash.create().update(key).digest() : key);
        for (let i = 0; i < pad.length; i++)
          pad[i] ^= 54;
        this.iHash.update(pad);
        this.oHash = hash.create();
        for (let i = 0; i < pad.length; i++)
          pad[i] ^= 54 ^ 92;
        this.oHash.update(pad);
        (0, utils_ts_1.clean)(pad);
      }
      update(buf) {
        (0, utils_ts_1.aexists)(this);
        this.iHash.update(buf);
        return this;
      }
      digestInto(out) {
        (0, utils_ts_1.aexists)(this);
        (0, utils_ts_1.abytes)(out, this.outputLen);
        this.finished = true;
        this.iHash.digestInto(out);
        this.oHash.update(out);
        this.oHash.digestInto(out);
        this.destroy();
      }
      digest() {
        const out = new Uint8Array(this.oHash.outputLen);
        this.digestInto(out);
        return out;
      }
      _cloneInto(to) {
        to || (to = Object.create(Object.getPrototypeOf(this), {}));
        const { oHash, iHash, finished, destroyed, blockLen, outputLen } = this;
        to = to;
        to.finished = finished;
        to.destroyed = destroyed;
        to.blockLen = blockLen;
        to.outputLen = outputLen;
        to.oHash = oHash._cloneInto(to.oHash);
        to.iHash = iHash._cloneInto(to.iHash);
        return to;
      }
      clone() {
        return this._cloneInto();
      }
      destroy() {
        this.destroyed = true;
        this.oHash.destroy();
        this.iHash.destroy();
      }
    };
    exports.HMAC = HMAC;
    var hmac = (hash, key, message2) => new HMAC(hash, key).update(message2).digest();
    exports.hmac = hmac;
    exports.hmac.create = (hash, key) => new HMAC(hash, key);
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/weierstrass.js
var require_weierstrass = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/weierstrass.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.DER = exports.DERErr = void 0;
    exports._splitEndoScalar = _splitEndoScalar;
    exports._normFnElement = _normFnElement;
    exports.weierstrassN = weierstrassN;
    exports.SWUFpSqrtRatio = SWUFpSqrtRatio;
    exports.mapToCurveSimpleSWU = mapToCurveSimpleSWU;
    exports.ecdh = ecdh;
    exports.ecdsa = ecdsa;
    exports.weierstrassPoints = weierstrassPoints;
    exports._legacyHelperEquat = _legacyHelperEquat;
    exports.weierstrass = weierstrass;
    var hmac_js_1 = require_hmac();
    var utils_1 = require_utils();
    var utils_ts_1 = require_utils2();
    var curve_ts_1 = require_curve();
    var modular_ts_1 = require_modular();
    var divNearest = (num, den) => (num + (num >= 0 ? den : -den) / _2n) / den;
    function _splitEndoScalar(k, basis, n) {
      const [[a1, b1], [a2, b2]] = basis;
      const c1 = divNearest(b2 * k, n);
      const c2 = divNearest(-b1 * k, n);
      let k1 = k - c1 * a1 - c2 * a2;
      let k2 = -c1 * b1 - c2 * b2;
      const k1neg = k1 < _0n;
      const k2neg = k2 < _0n;
      if (k1neg)
        k1 = -k1;
      if (k2neg)
        k2 = -k2;
      const MAX_NUM = (0, utils_ts_1.bitMask)(Math.ceil((0, utils_ts_1.bitLen)(n) / 2)) + _1n;
      if (k1 < _0n || k1 >= MAX_NUM || k2 < _0n || k2 >= MAX_NUM) {
        throw new Error("splitScalar (endomorphism): failed, k=" + k);
      }
      return { k1neg, k1, k2neg, k2 };
    }
    function validateSigFormat(format) {
      if (!["compact", "recovered", "der"].includes(format))
        throw new Error('Signature format must be "compact", "recovered", or "der"');
      return format;
    }
    function validateSigOpts(opts, def) {
      const optsn = {};
      for (let optName of Object.keys(def)) {
        optsn[optName] = opts[optName] === void 0 ? def[optName] : opts[optName];
      }
      (0, utils_ts_1._abool2)(optsn.lowS, "lowS");
      (0, utils_ts_1._abool2)(optsn.prehash, "prehash");
      if (optsn.format !== void 0)
        validateSigFormat(optsn.format);
      return optsn;
    }
    var DERErr = class extends Error {
      constructor(m = "") {
        super(m);
      }
    };
    exports.DERErr = DERErr;
    exports.DER = {
      // asn.1 DER encoding utils
      Err: DERErr,
      // Basic building block is TLV (Tag-Length-Value)
      _tlv: {
        encode: (tag, data) => {
          const { Err: E } = exports.DER;
          if (tag < 0 || tag > 256)
            throw new E("tlv.encode: wrong tag");
          if (data.length & 1)
            throw new E("tlv.encode: unpadded data");
          const dataLen = data.length / 2;
          const len = (0, utils_ts_1.numberToHexUnpadded)(dataLen);
          if (len.length / 2 & 128)
            throw new E("tlv.encode: long form length too big");
          const lenLen = dataLen > 127 ? (0, utils_ts_1.numberToHexUnpadded)(len.length / 2 | 128) : "";
          const t = (0, utils_ts_1.numberToHexUnpadded)(tag);
          return t + lenLen + len + data;
        },
        // v - value, l - left bytes (unparsed)
        decode(tag, data) {
          const { Err: E } = exports.DER;
          let pos = 0;
          if (tag < 0 || tag > 256)
            throw new E("tlv.encode: wrong tag");
          if (data.length < 2 || data[pos++] !== tag)
            throw new E("tlv.decode: wrong tlv");
          const first = data[pos++];
          const isLong = !!(first & 128);
          let length = 0;
          if (!isLong)
            length = first;
          else {
            const lenLen = first & 127;
            if (!lenLen)
              throw new E("tlv.decode(long): indefinite length not supported");
            if (lenLen > 4)
              throw new E("tlv.decode(long): byte length is too big");
            const lengthBytes = data.subarray(pos, pos + lenLen);
            if (lengthBytes.length !== lenLen)
              throw new E("tlv.decode: length bytes not complete");
            if (lengthBytes[0] === 0)
              throw new E("tlv.decode(long): zero leftmost byte");
            for (const b of lengthBytes)
              length = length << 8 | b;
            pos += lenLen;
            if (length < 128)
              throw new E("tlv.decode(long): not minimal encoding");
          }
          const v = data.subarray(pos, pos + length);
          if (v.length !== length)
            throw new E("tlv.decode: wrong value length");
          return { v, l: data.subarray(pos + length) };
        }
      },
      // https://crypto.stackexchange.com/a/57734 Leftmost bit of first byte is 'negative' flag,
      // since we always use positive integers here. It must always be empty:
      // - add zero byte if exists
      // - if next byte doesn't have a flag, leading zero is not allowed (minimal encoding)
      _int: {
        encode(num) {
          const { Err: E } = exports.DER;
          if (num < _0n)
            throw new E("integer: negative integers are not allowed");
          let hex = (0, utils_ts_1.numberToHexUnpadded)(num);
          if (Number.parseInt(hex[0], 16) & 8)
            hex = "00" + hex;
          if (hex.length & 1)
            throw new E("unexpected DER parsing assertion: unpadded hex");
          return hex;
        },
        decode(data) {
          const { Err: E } = exports.DER;
          if (data[0] & 128)
            throw new E("invalid signature integer: negative");
          if (data[0] === 0 && !(data[1] & 128))
            throw new E("invalid signature integer: unnecessary leading zero");
          return (0, utils_ts_1.bytesToNumberBE)(data);
        }
      },
      toSig(hex) {
        const { Err: E, _int: int, _tlv: tlv } = exports.DER;
        const data = (0, utils_ts_1.ensureBytes)("signature", hex);
        const { v: seqBytes, l: seqLeftBytes } = tlv.decode(48, data);
        if (seqLeftBytes.length)
          throw new E("invalid signature: left bytes after parsing");
        const { v: rBytes, l: rLeftBytes } = tlv.decode(2, seqBytes);
        const { v: sBytes, l: sLeftBytes } = tlv.decode(2, rLeftBytes);
        if (sLeftBytes.length)
          throw new E("invalid signature: left bytes after parsing");
        return { r: int.decode(rBytes), s: int.decode(sBytes) };
      },
      hexFromSig(sig) {
        const { _tlv: tlv, _int: int } = exports.DER;
        const rs = tlv.encode(2, int.encode(sig.r));
        const ss = tlv.encode(2, int.encode(sig.s));
        const seq = rs + ss;
        return tlv.encode(48, seq);
      }
    };
    var _0n = BigInt(0);
    var _1n = BigInt(1);
    var _2n = BigInt(2);
    var _3n = BigInt(3);
    var _4n = BigInt(4);
    function _normFnElement(Fn, key) {
      const { BYTES: expected } = Fn;
      let num;
      if (typeof key === "bigint") {
        num = key;
      } else {
        let bytes = (0, utils_ts_1.ensureBytes)("private key", key);
        try {
          num = Fn.fromBytes(bytes);
        } catch (error) {
          throw new Error(`invalid private key: expected ui8a of size ${expected}, got ${typeof key}`);
        }
      }
      if (!Fn.isValidNot0(num))
        throw new Error("invalid private key: out of range [1..N-1]");
      return num;
    }
    function weierstrassN(params, extraOpts = {}) {
      const validated = (0, curve_ts_1._createCurveFields)("weierstrass", params, extraOpts);
      const { Fp, Fn } = validated;
      let CURVE = validated.CURVE;
      const { h: cofactor, n: CURVE_ORDER } = CURVE;
      (0, utils_ts_1._validateObject)(extraOpts, {}, {
        allowInfinityPoint: "boolean",
        clearCofactor: "function",
        isTorsionFree: "function",
        fromBytes: "function",
        toBytes: "function",
        endo: "object",
        wrapPrivateKey: "boolean"
      });
      const { endo } = extraOpts;
      if (endo) {
        if (!Fp.is0(CURVE.a) || typeof endo.beta !== "bigint" || !Array.isArray(endo.basises)) {
          throw new Error('invalid endo: expected "beta": bigint and "basises": array');
        }
      }
      const lengths = getWLengths(Fp, Fn);
      function assertCompressionIsSupported() {
        if (!Fp.isOdd)
          throw new Error("compression is not supported: Field does not have .isOdd()");
      }
      function pointToBytes(_c, point, isCompressed) {
        const { x, y } = point.toAffine();
        const bx = Fp.toBytes(x);
        (0, utils_ts_1._abool2)(isCompressed, "isCompressed");
        if (isCompressed) {
          assertCompressionIsSupported();
          const hasEvenY = !Fp.isOdd(y);
          return (0, utils_ts_1.concatBytes)(pprefix(hasEvenY), bx);
        } else {
          return (0, utils_ts_1.concatBytes)(Uint8Array.of(4), bx, Fp.toBytes(y));
        }
      }
      function pointFromBytes(bytes) {
        (0, utils_ts_1._abytes2)(bytes, void 0, "Point");
        const { publicKey: comp, publicKeyUncompressed: uncomp } = lengths;
        const length = bytes.length;
        const head = bytes[0];
        const tail = bytes.subarray(1);
        if (length === comp && (head === 2 || head === 3)) {
          const x = Fp.fromBytes(tail);
          if (!Fp.isValid(x))
            throw new Error("bad point: is not on curve, wrong x");
          const y2 = weierstrassEquation(x);
          let y;
          try {
            y = Fp.sqrt(y2);
          } catch (sqrtError) {
            const err = sqrtError instanceof Error ? ": " + sqrtError.message : "";
            throw new Error("bad point: is not on curve, sqrt error" + err);
          }
          assertCompressionIsSupported();
          const isYOdd = Fp.isOdd(y);
          const isHeadOdd = (head & 1) === 1;
          if (isHeadOdd !== isYOdd)
            y = Fp.neg(y);
          return { x, y };
        } else if (length === uncomp && head === 4) {
          const L = Fp.BYTES;
          const x = Fp.fromBytes(tail.subarray(0, L));
          const y = Fp.fromBytes(tail.subarray(L, L * 2));
          if (!isValidXY(x, y))
            throw new Error("bad point: is not on curve");
          return { x, y };
        } else {
          throw new Error(`bad point: got length ${length}, expected compressed=${comp} or uncompressed=${uncomp}`);
        }
      }
      const encodePoint = extraOpts.toBytes || pointToBytes;
      const decodePoint = extraOpts.fromBytes || pointFromBytes;
      function weierstrassEquation(x) {
        const x2 = Fp.sqr(x);
        const x3 = Fp.mul(x2, x);
        return Fp.add(Fp.add(x3, Fp.mul(x, CURVE.a)), CURVE.b);
      }
      function isValidXY(x, y) {
        const left = Fp.sqr(y);
        const right = weierstrassEquation(x);
        return Fp.eql(left, right);
      }
      if (!isValidXY(CURVE.Gx, CURVE.Gy))
        throw new Error("bad curve params: generator point");
      const _4a3 = Fp.mul(Fp.pow(CURVE.a, _3n), _4n);
      const _27b2 = Fp.mul(Fp.sqr(CURVE.b), BigInt(27));
      if (Fp.is0(Fp.add(_4a3, _27b2)))
        throw new Error("bad curve params: a or b");
      function acoord(title, n, banZero = false) {
        if (!Fp.isValid(n) || banZero && Fp.is0(n))
          throw new Error(`bad point coordinate ${title}`);
        return n;
      }
      function aprjpoint(other) {
        if (!(other instanceof Point))
          throw new Error("ProjectivePoint expected");
      }
      function splitEndoScalarN(k) {
        if (!endo || !endo.basises)
          throw new Error("no endo");
        return _splitEndoScalar(k, endo.basises, Fn.ORDER);
      }
      const toAffineMemo = (0, utils_ts_1.memoized)((p, iz) => {
        const { X, Y, Z } = p;
        if (Fp.eql(Z, Fp.ONE))
          return { x: X, y: Y };
        const is0 = p.is0();
        if (iz == null)
          iz = is0 ? Fp.ONE : Fp.inv(Z);
        const x = Fp.mul(X, iz);
        const y = Fp.mul(Y, iz);
        const zz = Fp.mul(Z, iz);
        if (is0)
          return { x: Fp.ZERO, y: Fp.ZERO };
        if (!Fp.eql(zz, Fp.ONE))
          throw new Error("invZ was invalid");
        return { x, y };
      });
      const assertValidMemo = (0, utils_ts_1.memoized)((p) => {
        if (p.is0()) {
          if (extraOpts.allowInfinityPoint && !Fp.is0(p.Y))
            return;
          throw new Error("bad point: ZERO");
        }
        const { x, y } = p.toAffine();
        if (!Fp.isValid(x) || !Fp.isValid(y))
          throw new Error("bad point: x or y not field elements");
        if (!isValidXY(x, y))
          throw new Error("bad point: equation left != right");
        if (!p.isTorsionFree())
          throw new Error("bad point: not in prime-order subgroup");
        return true;
      });
      function finishEndo(endoBeta, k1p, k2p, k1neg, k2neg) {
        k2p = new Point(Fp.mul(k2p.X, endoBeta), k2p.Y, k2p.Z);
        k1p = (0, curve_ts_1.negateCt)(k1neg, k1p);
        k2p = (0, curve_ts_1.negateCt)(k2neg, k2p);
        return k1p.add(k2p);
      }
      class Point {
        /** Does NOT validate if the point is valid. Use `.assertValidity()`. */
        constructor(X, Y, Z) {
          this.X = acoord("x", X);
          this.Y = acoord("y", Y, true);
          this.Z = acoord("z", Z);
          Object.freeze(this);
        }
        static CURVE() {
          return CURVE;
        }
        /** Does NOT validate if the point is valid. Use `.assertValidity()`. */
        static fromAffine(p) {
          const { x, y } = p || {};
          if (!p || !Fp.isValid(x) || !Fp.isValid(y))
            throw new Error("invalid affine point");
          if (p instanceof Point)
            throw new Error("projective point not allowed");
          if (Fp.is0(x) && Fp.is0(y))
            return Point.ZERO;
          return new Point(x, y, Fp.ONE);
        }
        static fromBytes(bytes) {
          const P = Point.fromAffine(decodePoint((0, utils_ts_1._abytes2)(bytes, void 0, "point")));
          P.assertValidity();
          return P;
        }
        static fromHex(hex) {
          return Point.fromBytes((0, utils_ts_1.ensureBytes)("pointHex", hex));
        }
        get x() {
          return this.toAffine().x;
        }
        get y() {
          return this.toAffine().y;
        }
        /**
         *
         * @param windowSize
         * @param isLazy true will defer table computation until the first multiplication
         * @returns
         */
        precompute(windowSize = 8, isLazy = true) {
          wnaf.createCache(this, windowSize);
          if (!isLazy)
            this.multiply(_3n);
          return this;
        }
        // TODO: return `this`
        /** A point on curve is valid if it conforms to equation. */
        assertValidity() {
          assertValidMemo(this);
        }
        hasEvenY() {
          const { y } = this.toAffine();
          if (!Fp.isOdd)
            throw new Error("Field doesn't support isOdd");
          return !Fp.isOdd(y);
        }
        /** Compare one point to another. */
        equals(other) {
          aprjpoint(other);
          const { X: X1, Y: Y1, Z: Z1 } = this;
          const { X: X2, Y: Y2, Z: Z2 } = other;
          const U1 = Fp.eql(Fp.mul(X1, Z2), Fp.mul(X2, Z1));
          const U2 = Fp.eql(Fp.mul(Y1, Z2), Fp.mul(Y2, Z1));
          return U1 && U2;
        }
        /** Flips point to one corresponding to (x, -y) in Affine coordinates. */
        negate() {
          return new Point(this.X, Fp.neg(this.Y), this.Z);
        }
        // Renes-Costello-Batina exception-free doubling formula.
        // There is 30% faster Jacobian formula, but it is not complete.
        // https://eprint.iacr.org/2015/1060, algorithm 3
        // Cost: 8M + 3S + 3*a + 2*b3 + 15add.
        double() {
          const { a, b } = CURVE;
          const b3 = Fp.mul(b, _3n);
          const { X: X1, Y: Y1, Z: Z1 } = this;
          let X3 = Fp.ZERO, Y3 = Fp.ZERO, Z3 = Fp.ZERO;
          let t0 = Fp.mul(X1, X1);
          let t1 = Fp.mul(Y1, Y1);
          let t2 = Fp.mul(Z1, Z1);
          let t3 = Fp.mul(X1, Y1);
          t3 = Fp.add(t3, t3);
          Z3 = Fp.mul(X1, Z1);
          Z3 = Fp.add(Z3, Z3);
          X3 = Fp.mul(a, Z3);
          Y3 = Fp.mul(b3, t2);
          Y3 = Fp.add(X3, Y3);
          X3 = Fp.sub(t1, Y3);
          Y3 = Fp.add(t1, Y3);
          Y3 = Fp.mul(X3, Y3);
          X3 = Fp.mul(t3, X3);
          Z3 = Fp.mul(b3, Z3);
          t2 = Fp.mul(a, t2);
          t3 = Fp.sub(t0, t2);
          t3 = Fp.mul(a, t3);
          t3 = Fp.add(t3, Z3);
          Z3 = Fp.add(t0, t0);
          t0 = Fp.add(Z3, t0);
          t0 = Fp.add(t0, t2);
          t0 = Fp.mul(t0, t3);
          Y3 = Fp.add(Y3, t0);
          t2 = Fp.mul(Y1, Z1);
          t2 = Fp.add(t2, t2);
          t0 = Fp.mul(t2, t3);
          X3 = Fp.sub(X3, t0);
          Z3 = Fp.mul(t2, t1);
          Z3 = Fp.add(Z3, Z3);
          Z3 = Fp.add(Z3, Z3);
          return new Point(X3, Y3, Z3);
        }
        // Renes-Costello-Batina exception-free addition formula.
        // There is 30% faster Jacobian formula, but it is not complete.
        // https://eprint.iacr.org/2015/1060, algorithm 1
        // Cost: 12M + 0S + 3*a + 3*b3 + 23add.
        add(other) {
          aprjpoint(other);
          const { X: X1, Y: Y1, Z: Z1 } = this;
          const { X: X2, Y: Y2, Z: Z2 } = other;
          let X3 = Fp.ZERO, Y3 = Fp.ZERO, Z3 = Fp.ZERO;
          const a = CURVE.a;
          const b3 = Fp.mul(CURVE.b, _3n);
          let t0 = Fp.mul(X1, X2);
          let t1 = Fp.mul(Y1, Y2);
          let t2 = Fp.mul(Z1, Z2);
          let t3 = Fp.add(X1, Y1);
          let t4 = Fp.add(X2, Y2);
          t3 = Fp.mul(t3, t4);
          t4 = Fp.add(t0, t1);
          t3 = Fp.sub(t3, t4);
          t4 = Fp.add(X1, Z1);
          let t5 = Fp.add(X2, Z2);
          t4 = Fp.mul(t4, t5);
          t5 = Fp.add(t0, t2);
          t4 = Fp.sub(t4, t5);
          t5 = Fp.add(Y1, Z1);
          X3 = Fp.add(Y2, Z2);
          t5 = Fp.mul(t5, X3);
          X3 = Fp.add(t1, t2);
          t5 = Fp.sub(t5, X3);
          Z3 = Fp.mul(a, t4);
          X3 = Fp.mul(b3, t2);
          Z3 = Fp.add(X3, Z3);
          X3 = Fp.sub(t1, Z3);
          Z3 = Fp.add(t1, Z3);
          Y3 = Fp.mul(X3, Z3);
          t1 = Fp.add(t0, t0);
          t1 = Fp.add(t1, t0);
          t2 = Fp.mul(a, t2);
          t4 = Fp.mul(b3, t4);
          t1 = Fp.add(t1, t2);
          t2 = Fp.sub(t0, t2);
          t2 = Fp.mul(a, t2);
          t4 = Fp.add(t4, t2);
          t0 = Fp.mul(t1, t4);
          Y3 = Fp.add(Y3, t0);
          t0 = Fp.mul(t5, t4);
          X3 = Fp.mul(t3, X3);
          X3 = Fp.sub(X3, t0);
          t0 = Fp.mul(t3, t1);
          Z3 = Fp.mul(t5, Z3);
          Z3 = Fp.add(Z3, t0);
          return new Point(X3, Y3, Z3);
        }
        subtract(other) {
          return this.add(other.negate());
        }
        is0() {
          return this.equals(Point.ZERO);
        }
        /**
         * Constant time multiplication.
         * Uses wNAF method. Windowed method may be 10% faster,
         * but takes 2x longer to generate and consumes 2x memory.
         * Uses precomputes when available.
         * Uses endomorphism for Koblitz curves.
         * @param scalar by which the point would be multiplied
         * @returns New point
         */
        multiply(scalar) {
          const { endo: endo2 } = extraOpts;
          if (!Fn.isValidNot0(scalar))
            throw new Error("invalid scalar: out of range");
          let point, fake;
          const mul = (n) => wnaf.cached(this, n, (p) => (0, curve_ts_1.normalizeZ)(Point, p));
          if (endo2) {
            const { k1neg, k1, k2neg, k2 } = splitEndoScalarN(scalar);
            const { p: k1p, f: k1f } = mul(k1);
            const { p: k2p, f: k2f } = mul(k2);
            fake = k1f.add(k2f);
            point = finishEndo(endo2.beta, k1p, k2p, k1neg, k2neg);
          } else {
            const { p, f } = mul(scalar);
            point = p;
            fake = f;
          }
          return (0, curve_ts_1.normalizeZ)(Point, [point, fake])[0];
        }
        /**
         * Non-constant-time multiplication. Uses double-and-add algorithm.
         * It's faster, but should only be used when you don't care about
         * an exposed secret key e.g. sig verification, which works over *public* keys.
         */
        multiplyUnsafe(sc) {
          const { endo: endo2 } = extraOpts;
          const p = this;
          if (!Fn.isValid(sc))
            throw new Error("invalid scalar: out of range");
          if (sc === _0n || p.is0())
            return Point.ZERO;
          if (sc === _1n)
            return p;
          if (wnaf.hasCache(this))
            return this.multiply(sc);
          if (endo2) {
            const { k1neg, k1, k2neg, k2 } = splitEndoScalarN(sc);
            const { p1, p2 } = (0, curve_ts_1.mulEndoUnsafe)(Point, p, k1, k2);
            return finishEndo(endo2.beta, p1, p2, k1neg, k2neg);
          } else {
            return wnaf.unsafe(p, sc);
          }
        }
        multiplyAndAddUnsafe(Q, a, b) {
          const sum = this.multiplyUnsafe(a).add(Q.multiplyUnsafe(b));
          return sum.is0() ? void 0 : sum;
        }
        /**
         * Converts Projective point to affine (x, y) coordinates.
         * @param invertedZ Z^-1 (inverted zero) - optional, precomputation is useful for invertBatch
         */
        toAffine(invertedZ) {
          return toAffineMemo(this, invertedZ);
        }
        /**
         * Checks whether Point is free of torsion elements (is in prime subgroup).
         * Always torsion-free for cofactor=1 curves.
         */
        isTorsionFree() {
          const { isTorsionFree } = extraOpts;
          if (cofactor === _1n)
            return true;
          if (isTorsionFree)
            return isTorsionFree(Point, this);
          return wnaf.unsafe(this, CURVE_ORDER).is0();
        }
        clearCofactor() {
          const { clearCofactor } = extraOpts;
          if (cofactor === _1n)
            return this;
          if (clearCofactor)
            return clearCofactor(Point, this);
          return this.multiplyUnsafe(cofactor);
        }
        isSmallOrder() {
          return this.multiplyUnsafe(cofactor).is0();
        }
        toBytes(isCompressed = true) {
          (0, utils_ts_1._abool2)(isCompressed, "isCompressed");
          this.assertValidity();
          return encodePoint(Point, this, isCompressed);
        }
        toHex(isCompressed = true) {
          return (0, utils_ts_1.bytesToHex)(this.toBytes(isCompressed));
        }
        toString() {
          return `<Point ${this.is0() ? "ZERO" : this.toHex()}>`;
        }
        // TODO: remove
        get px() {
          return this.X;
        }
        get py() {
          return this.X;
        }
        get pz() {
          return this.Z;
        }
        toRawBytes(isCompressed = true) {
          return this.toBytes(isCompressed);
        }
        _setWindowSize(windowSize) {
          this.precompute(windowSize);
        }
        static normalizeZ(points) {
          return (0, curve_ts_1.normalizeZ)(Point, points);
        }
        static msm(points, scalars) {
          return (0, curve_ts_1.pippenger)(Point, Fn, points, scalars);
        }
        static fromPrivateKey(privateKey) {
          return Point.BASE.multiply(_normFnElement(Fn, privateKey));
        }
      }
      Point.BASE = new Point(CURVE.Gx, CURVE.Gy, Fp.ONE);
      Point.ZERO = new Point(Fp.ZERO, Fp.ONE, Fp.ZERO);
      Point.Fp = Fp;
      Point.Fn = Fn;
      const bits = Fn.BITS;
      const wnaf = new curve_ts_1.wNAF(Point, extraOpts.endo ? Math.ceil(bits / 2) : bits);
      Point.BASE.precompute(8);
      return Point;
    }
    function pprefix(hasEvenY) {
      return Uint8Array.of(hasEvenY ? 2 : 3);
    }
    function SWUFpSqrtRatio(Fp, Z) {
      const q = Fp.ORDER;
      let l = _0n;
      for (let o = q - _1n; o % _2n === _0n; o /= _2n)
        l += _1n;
      const c1 = l;
      const _2n_pow_c1_1 = _2n << c1 - _1n - _1n;
      const _2n_pow_c1 = _2n_pow_c1_1 * _2n;
      const c2 = (q - _1n) / _2n_pow_c1;
      const c3 = (c2 - _1n) / _2n;
      const c4 = _2n_pow_c1 - _1n;
      const c5 = _2n_pow_c1_1;
      const c6 = Fp.pow(Z, c2);
      const c7 = Fp.pow(Z, (c2 + _1n) / _2n);
      let sqrtRatio = (u, v) => {
        let tv1 = c6;
        let tv2 = Fp.pow(v, c4);
        let tv3 = Fp.sqr(tv2);
        tv3 = Fp.mul(tv3, v);
        let tv5 = Fp.mul(u, tv3);
        tv5 = Fp.pow(tv5, c3);
        tv5 = Fp.mul(tv5, tv2);
        tv2 = Fp.mul(tv5, v);
        tv3 = Fp.mul(tv5, u);
        let tv4 = Fp.mul(tv3, tv2);
        tv5 = Fp.pow(tv4, c5);
        let isQR = Fp.eql(tv5, Fp.ONE);
        tv2 = Fp.mul(tv3, c7);
        tv5 = Fp.mul(tv4, tv1);
        tv3 = Fp.cmov(tv2, tv3, isQR);
        tv4 = Fp.cmov(tv5, tv4, isQR);
        for (let i = c1; i > _1n; i--) {
          let tv52 = i - _2n;
          tv52 = _2n << tv52 - _1n;
          let tvv5 = Fp.pow(tv4, tv52);
          const e1 = Fp.eql(tvv5, Fp.ONE);
          tv2 = Fp.mul(tv3, tv1);
          tv1 = Fp.mul(tv1, tv1);
          tvv5 = Fp.mul(tv4, tv1);
          tv3 = Fp.cmov(tv2, tv3, e1);
          tv4 = Fp.cmov(tvv5, tv4, e1);
        }
        return { isValid: isQR, value: tv3 };
      };
      if (Fp.ORDER % _4n === _3n) {
        const c12 = (Fp.ORDER - _3n) / _4n;
        const c22 = Fp.sqrt(Fp.neg(Z));
        sqrtRatio = (u, v) => {
          let tv1 = Fp.sqr(v);
          const tv2 = Fp.mul(u, v);
          tv1 = Fp.mul(tv1, tv2);
          let y1 = Fp.pow(tv1, c12);
          y1 = Fp.mul(y1, tv2);
          const y2 = Fp.mul(y1, c22);
          const tv3 = Fp.mul(Fp.sqr(y1), v);
          const isQR = Fp.eql(tv3, u);
          let y = Fp.cmov(y2, y1, isQR);
          return { isValid: isQR, value: y };
        };
      }
      return sqrtRatio;
    }
    function mapToCurveSimpleSWU(Fp, opts) {
      (0, modular_ts_1.validateField)(Fp);
      const { A, B, Z } = opts;
      if (!Fp.isValid(A) || !Fp.isValid(B) || !Fp.isValid(Z))
        throw new Error("mapToCurveSimpleSWU: invalid opts");
      const sqrtRatio = SWUFpSqrtRatio(Fp, Z);
      if (!Fp.isOdd)
        throw new Error("Field does not have .isOdd()");
      return (u) => {
        let tv1, tv2, tv3, tv4, tv5, tv6, x, y;
        tv1 = Fp.sqr(u);
        tv1 = Fp.mul(tv1, Z);
        tv2 = Fp.sqr(tv1);
        tv2 = Fp.add(tv2, tv1);
        tv3 = Fp.add(tv2, Fp.ONE);
        tv3 = Fp.mul(tv3, B);
        tv4 = Fp.cmov(Z, Fp.neg(tv2), !Fp.eql(tv2, Fp.ZERO));
        tv4 = Fp.mul(tv4, A);
        tv2 = Fp.sqr(tv3);
        tv6 = Fp.sqr(tv4);
        tv5 = Fp.mul(tv6, A);
        tv2 = Fp.add(tv2, tv5);
        tv2 = Fp.mul(tv2, tv3);
        tv6 = Fp.mul(tv6, tv4);
        tv5 = Fp.mul(tv6, B);
        tv2 = Fp.add(tv2, tv5);
        x = Fp.mul(tv1, tv3);
        const { isValid, value } = sqrtRatio(tv2, tv6);
        y = Fp.mul(tv1, u);
        y = Fp.mul(y, value);
        x = Fp.cmov(x, tv3, isValid);
        y = Fp.cmov(y, value, isValid);
        const e1 = Fp.isOdd(u) === Fp.isOdd(y);
        y = Fp.cmov(Fp.neg(y), y, e1);
        const tv4_inv = (0, modular_ts_1.FpInvertBatch)(Fp, [tv4], true)[0];
        x = Fp.mul(x, tv4_inv);
        return { x, y };
      };
    }
    function getWLengths(Fp, Fn) {
      return {
        secretKey: Fn.BYTES,
        publicKey: 1 + Fp.BYTES,
        publicKeyUncompressed: 1 + 2 * Fp.BYTES,
        publicKeyHasPrefix: true,
        signature: 2 * Fn.BYTES
      };
    }
    function ecdh(Point, ecdhOpts = {}) {
      const { Fn } = Point;
      const randomBytes_ = ecdhOpts.randomBytes || utils_ts_1.randomBytes;
      const lengths = Object.assign(getWLengths(Point.Fp, Fn), { seed: (0, modular_ts_1.getMinHashLength)(Fn.ORDER) });
      function isValidSecretKey(secretKey) {
        try {
          return !!_normFnElement(Fn, secretKey);
        } catch (error) {
          return false;
        }
      }
      function isValidPublicKey(publicKey, isCompressed) {
        const { publicKey: comp, publicKeyUncompressed } = lengths;
        try {
          const l = publicKey.length;
          if (isCompressed === true && l !== comp)
            return false;
          if (isCompressed === false && l !== publicKeyUncompressed)
            return false;
          return !!Point.fromBytes(publicKey);
        } catch (error) {
          return false;
        }
      }
      function randomSecretKey(seed = randomBytes_(lengths.seed)) {
        return (0, modular_ts_1.mapHashToField)((0, utils_ts_1._abytes2)(seed, lengths.seed, "seed"), Fn.ORDER);
      }
      function getPublicKey(secretKey, isCompressed = true) {
        return Point.BASE.multiply(_normFnElement(Fn, secretKey)).toBytes(isCompressed);
      }
      function keygen(seed) {
        const secretKey = randomSecretKey(seed);
        return { secretKey, publicKey: getPublicKey(secretKey) };
      }
      function isProbPub(item) {
        if (typeof item === "bigint")
          return false;
        if (item instanceof Point)
          return true;
        const { secretKey, publicKey, publicKeyUncompressed } = lengths;
        if (Fn.allowedLengths || secretKey === publicKey)
          return void 0;
        const l = (0, utils_ts_1.ensureBytes)("key", item).length;
        return l === publicKey || l === publicKeyUncompressed;
      }
      function getSharedSecret(secretKeyA, publicKeyB, isCompressed = true) {
        if (isProbPub(secretKeyA) === true)
          throw new Error("first arg must be private key");
        if (isProbPub(publicKeyB) === false)
          throw new Error("second arg must be public key");
        const s = _normFnElement(Fn, secretKeyA);
        const b = Point.fromHex(publicKeyB);
        return b.multiply(s).toBytes(isCompressed);
      }
      const utils = {
        isValidSecretKey,
        isValidPublicKey,
        randomSecretKey,
        // TODO: remove
        isValidPrivateKey: isValidSecretKey,
        randomPrivateKey: randomSecretKey,
        normPrivateKeyToScalar: (key) => _normFnElement(Fn, key),
        precompute(windowSize = 8, point = Point.BASE) {
          return point.precompute(windowSize, false);
        }
      };
      return Object.freeze({ getPublicKey, getSharedSecret, keygen, Point, utils, lengths });
    }
    function ecdsa(Point, hash, ecdsaOpts = {}) {
      (0, utils_1.ahash)(hash);
      (0, utils_ts_1._validateObject)(ecdsaOpts, {}, {
        hmac: "function",
        lowS: "boolean",
        randomBytes: "function",
        bits2int: "function",
        bits2int_modN: "function"
      });
      const randomBytes = ecdsaOpts.randomBytes || utils_ts_1.randomBytes;
      const hmac = ecdsaOpts.hmac || ((key, ...msgs) => (0, hmac_js_1.hmac)(hash, key, (0, utils_ts_1.concatBytes)(...msgs)));
      const { Fp, Fn } = Point;
      const { ORDER: CURVE_ORDER, BITS: fnBits } = Fn;
      const { keygen, getPublicKey, getSharedSecret, utils, lengths } = ecdh(Point, ecdsaOpts);
      const defaultSigOpts = {
        prehash: false,
        lowS: typeof ecdsaOpts.lowS === "boolean" ? ecdsaOpts.lowS : false,
        format: void 0,
        //'compact' as ECDSASigFormat,
        extraEntropy: false
      };
      const defaultSigOpts_format = "compact";
      function isBiggerThanHalfOrder(number) {
        const HALF = CURVE_ORDER >> _1n;
        return number > HALF;
      }
      function validateRS(title, num) {
        if (!Fn.isValidNot0(num))
          throw new Error(`invalid signature ${title}: out of range 1..Point.Fn.ORDER`);
        return num;
      }
      function validateSigLength(bytes, format) {
        validateSigFormat(format);
        const size = lengths.signature;
        const sizer = format === "compact" ? size : format === "recovered" ? size + 1 : void 0;
        return (0, utils_ts_1._abytes2)(bytes, sizer, `${format} signature`);
      }
      class Signature {
        constructor(r, s, recovery) {
          this.r = validateRS("r", r);
          this.s = validateRS("s", s);
          if (recovery != null)
            this.recovery = recovery;
          Object.freeze(this);
        }
        static fromBytes(bytes, format = defaultSigOpts_format) {
          validateSigLength(bytes, format);
          let recid;
          if (format === "der") {
            const { r: r2, s: s2 } = exports.DER.toSig((0, utils_ts_1._abytes2)(bytes));
            return new Signature(r2, s2);
          }
          if (format === "recovered") {
            recid = bytes[0];
            format = "compact";
            bytes = bytes.subarray(1);
          }
          const L = Fn.BYTES;
          const r = bytes.subarray(0, L);
          const s = bytes.subarray(L, L * 2);
          return new Signature(Fn.fromBytes(r), Fn.fromBytes(s), recid);
        }
        static fromHex(hex, format) {
          return this.fromBytes((0, utils_ts_1.hexToBytes)(hex), format);
        }
        addRecoveryBit(recovery) {
          return new Signature(this.r, this.s, recovery);
        }
        recoverPublicKey(messageHash) {
          const FIELD_ORDER = Fp.ORDER;
          const { r, s, recovery: rec } = this;
          if (rec == null || ![0, 1, 2, 3].includes(rec))
            throw new Error("recovery id invalid");
          const hasCofactor = CURVE_ORDER * _2n < FIELD_ORDER;
          if (hasCofactor && rec > 1)
            throw new Error("recovery id is ambiguous for h>1 curve");
          const radj = rec === 2 || rec === 3 ? r + CURVE_ORDER : r;
          if (!Fp.isValid(radj))
            throw new Error("recovery id 2 or 3 invalid");
          const x = Fp.toBytes(radj);
          const R = Point.fromBytes((0, utils_ts_1.concatBytes)(pprefix((rec & 1) === 0), x));
          const ir = Fn.inv(radj);
          const h = bits2int_modN((0, utils_ts_1.ensureBytes)("msgHash", messageHash));
          const u1 = Fn.create(-h * ir);
          const u2 = Fn.create(s * ir);
          const Q = Point.BASE.multiplyUnsafe(u1).add(R.multiplyUnsafe(u2));
          if (Q.is0())
            throw new Error("point at infinify");
          Q.assertValidity();
          return Q;
        }
        // Signatures should be low-s, to prevent malleability.
        hasHighS() {
          return isBiggerThanHalfOrder(this.s);
        }
        toBytes(format = defaultSigOpts_format) {
          validateSigFormat(format);
          if (format === "der")
            return (0, utils_ts_1.hexToBytes)(exports.DER.hexFromSig(this));
          const r = Fn.toBytes(this.r);
          const s = Fn.toBytes(this.s);
          if (format === "recovered") {
            if (this.recovery == null)
              throw new Error("recovery bit must be present");
            return (0, utils_ts_1.concatBytes)(Uint8Array.of(this.recovery), r, s);
          }
          return (0, utils_ts_1.concatBytes)(r, s);
        }
        toHex(format) {
          return (0, utils_ts_1.bytesToHex)(this.toBytes(format));
        }
        // TODO: remove
        assertValidity() {
        }
        static fromCompact(hex) {
          return Signature.fromBytes((0, utils_ts_1.ensureBytes)("sig", hex), "compact");
        }
        static fromDER(hex) {
          return Signature.fromBytes((0, utils_ts_1.ensureBytes)("sig", hex), "der");
        }
        normalizeS() {
          return this.hasHighS() ? new Signature(this.r, Fn.neg(this.s), this.recovery) : this;
        }
        toDERRawBytes() {
          return this.toBytes("der");
        }
        toDERHex() {
          return (0, utils_ts_1.bytesToHex)(this.toBytes("der"));
        }
        toCompactRawBytes() {
          return this.toBytes("compact");
        }
        toCompactHex() {
          return (0, utils_ts_1.bytesToHex)(this.toBytes("compact"));
        }
      }
      const bits2int = ecdsaOpts.bits2int || function bits2int_def(bytes) {
        if (bytes.length > 8192)
          throw new Error("input is too large");
        const num = (0, utils_ts_1.bytesToNumberBE)(bytes);
        const delta = bytes.length * 8 - fnBits;
        return delta > 0 ? num >> BigInt(delta) : num;
      };
      const bits2int_modN = ecdsaOpts.bits2int_modN || function bits2int_modN_def(bytes) {
        return Fn.create(bits2int(bytes));
      };
      const ORDER_MASK = (0, utils_ts_1.bitMask)(fnBits);
      function int2octets(num) {
        (0, utils_ts_1.aInRange)("num < 2^" + fnBits, num, _0n, ORDER_MASK);
        return Fn.toBytes(num);
      }
      function validateMsgAndHash(message2, prehash) {
        (0, utils_ts_1._abytes2)(message2, void 0, "message");
        return prehash ? (0, utils_ts_1._abytes2)(hash(message2), void 0, "prehashed message") : message2;
      }
      function prepSig(message2, privateKey, opts) {
        if (["recovered", "canonical"].some((k) => k in opts))
          throw new Error("sign() legacy options not supported");
        const { lowS, prehash, extraEntropy } = validateSigOpts(opts, defaultSigOpts);
        message2 = validateMsgAndHash(message2, prehash);
        const h1int = bits2int_modN(message2);
        const d = _normFnElement(Fn, privateKey);
        const seedArgs = [int2octets(d), int2octets(h1int)];
        if (extraEntropy != null && extraEntropy !== false) {
          const e = extraEntropy === true ? randomBytes(lengths.secretKey) : extraEntropy;
          seedArgs.push((0, utils_ts_1.ensureBytes)("extraEntropy", e));
        }
        const seed = (0, utils_ts_1.concatBytes)(...seedArgs);
        const m = h1int;
        function k2sig(kBytes) {
          const k = bits2int(kBytes);
          if (!Fn.isValidNot0(k))
            return;
          const ik = Fn.inv(k);
          const q = Point.BASE.multiply(k).toAffine();
          const r = Fn.create(q.x);
          if (r === _0n)
            return;
          const s = Fn.create(ik * Fn.create(m + r * d));
          if (s === _0n)
            return;
          let recovery = (q.x === r ? 0 : 2) | Number(q.y & _1n);
          let normS = s;
          if (lowS && isBiggerThanHalfOrder(s)) {
            normS = Fn.neg(s);
            recovery ^= 1;
          }
          return new Signature(r, normS, recovery);
        }
        return { seed, k2sig };
      }
      function sign(message2, secretKey, opts = {}) {
        message2 = (0, utils_ts_1.ensureBytes)("message", message2);
        const { seed, k2sig } = prepSig(message2, secretKey, opts);
        const drbg = (0, utils_ts_1.createHmacDrbg)(hash.outputLen, Fn.BYTES, hmac);
        const sig = drbg(seed, k2sig);
        return sig;
      }
      function tryParsingSig(sg) {
        let sig = void 0;
        const isHex = typeof sg === "string" || (0, utils_ts_1.isBytes)(sg);
        const isObj = !isHex && sg !== null && typeof sg === "object" && typeof sg.r === "bigint" && typeof sg.s === "bigint";
        if (!isHex && !isObj)
          throw new Error("invalid signature, expected Uint8Array, hex string or Signature instance");
        if (isObj) {
          sig = new Signature(sg.r, sg.s);
        } else if (isHex) {
          try {
            sig = Signature.fromBytes((0, utils_ts_1.ensureBytes)("sig", sg), "der");
          } catch (derError) {
            if (!(derError instanceof exports.DER.Err))
              throw derError;
          }
          if (!sig) {
            try {
              sig = Signature.fromBytes((0, utils_ts_1.ensureBytes)("sig", sg), "compact");
            } catch (error) {
              return false;
            }
          }
        }
        if (!sig)
          return false;
        return sig;
      }
      function verify(signature2, message2, publicKey, opts = {}) {
        const { lowS, prehash, format } = validateSigOpts(opts, defaultSigOpts);
        publicKey = (0, utils_ts_1.ensureBytes)("publicKey", publicKey);
        message2 = validateMsgAndHash((0, utils_ts_1.ensureBytes)("message", message2), prehash);
        if ("strict" in opts)
          throw new Error("options.strict was renamed to lowS");
        const sig = format === void 0 ? tryParsingSig(signature2) : Signature.fromBytes((0, utils_ts_1.ensureBytes)("sig", signature2), format);
        if (sig === false)
          return false;
        try {
          const P = Point.fromBytes(publicKey);
          if (lowS && sig.hasHighS())
            return false;
          const { r, s } = sig;
          const h = bits2int_modN(message2);
          const is = Fn.inv(s);
          const u1 = Fn.create(h * is);
          const u2 = Fn.create(r * is);
          const R = Point.BASE.multiplyUnsafe(u1).add(P.multiplyUnsafe(u2));
          if (R.is0())
            return false;
          const v = Fn.create(R.x);
          return v === r;
        } catch (e) {
          return false;
        }
      }
      function recoverPublicKey(signature2, message2, opts = {}) {
        const { prehash } = validateSigOpts(opts, defaultSigOpts);
        message2 = validateMsgAndHash(message2, prehash);
        return Signature.fromBytes(signature2, "recovered").recoverPublicKey(message2).toBytes();
      }
      return Object.freeze({
        keygen,
        getPublicKey,
        getSharedSecret,
        utils,
        lengths,
        Point,
        sign,
        verify,
        recoverPublicKey,
        Signature,
        hash
      });
    }
    function weierstrassPoints(c) {
      const { CURVE, curveOpts } = _weierstrass_legacy_opts_to_new(c);
      const Point = weierstrassN(CURVE, curveOpts);
      return _weierstrass_new_output_to_legacy(c, Point);
    }
    function _weierstrass_legacy_opts_to_new(c) {
      const CURVE = {
        a: c.a,
        b: c.b,
        p: c.Fp.ORDER,
        n: c.n,
        h: c.h,
        Gx: c.Gx,
        Gy: c.Gy
      };
      const Fp = c.Fp;
      let allowedLengths = c.allowedPrivateKeyLengths ? Array.from(new Set(c.allowedPrivateKeyLengths.map((l) => Math.ceil(l / 2)))) : void 0;
      const Fn = (0, modular_ts_1.Field)(CURVE.n, {
        BITS: c.nBitLength,
        allowedLengths,
        modFromBytes: c.wrapPrivateKey
      });
      const curveOpts = {
        Fp,
        Fn,
        allowInfinityPoint: c.allowInfinityPoint,
        endo: c.endo,
        isTorsionFree: c.isTorsionFree,
        clearCofactor: c.clearCofactor,
        fromBytes: c.fromBytes,
        toBytes: c.toBytes
      };
      return { CURVE, curveOpts };
    }
    function _ecdsa_legacy_opts_to_new(c) {
      const { CURVE, curveOpts } = _weierstrass_legacy_opts_to_new(c);
      const ecdsaOpts = {
        hmac: c.hmac,
        randomBytes: c.randomBytes,
        lowS: c.lowS,
        bits2int: c.bits2int,
        bits2int_modN: c.bits2int_modN
      };
      return { CURVE, curveOpts, hash: c.hash, ecdsaOpts };
    }
    function _legacyHelperEquat(Fp, a, b) {
      function weierstrassEquation(x) {
        const x2 = Fp.sqr(x);
        const x3 = Fp.mul(x2, x);
        return Fp.add(Fp.add(x3, Fp.mul(x, a)), b);
      }
      return weierstrassEquation;
    }
    function _weierstrass_new_output_to_legacy(c, Point) {
      const { Fp, Fn } = Point;
      function isWithinCurveOrder(num) {
        return (0, utils_ts_1.inRange)(num, _1n, Fn.ORDER);
      }
      const weierstrassEquation = _legacyHelperEquat(Fp, c.a, c.b);
      return Object.assign({}, {
        CURVE: c,
        Point,
        ProjectivePoint: Point,
        normPrivateKeyToScalar: (key) => _normFnElement(Fn, key),
        weierstrassEquation,
        isWithinCurveOrder
      });
    }
    function _ecdsa_new_output_to_legacy(c, _ecdsa) {
      const Point = _ecdsa.Point;
      return Object.assign({}, _ecdsa, {
        ProjectivePoint: Point,
        CURVE: Object.assign({}, c, (0, modular_ts_1.nLength)(Point.Fn.ORDER, Point.Fn.BITS))
      });
    }
    function weierstrass(c) {
      const { CURVE, curveOpts, hash, ecdsaOpts } = _ecdsa_legacy_opts_to_new(c);
      const Point = weierstrassN(CURVE, curveOpts);
      const signs = ecdsa(Point, hash, ecdsaOpts);
      return _ecdsa_new_output_to_legacy(c, signs);
    }
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/bls.js
var require_bls = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/bls.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.bls = bls2;
    var utils_ts_1 = require_utils2();
    var curve_ts_1 = require_curve();
    var hash_to_curve_ts_1 = require_hash_to_curve();
    var modular_ts_1 = require_modular();
    var weierstrass_ts_1 = require_weierstrass();
    var _0n = BigInt(0);
    var _1n = BigInt(1);
    var _2n = BigInt(2);
    var _3n = BigInt(3);
    function NAfDecomposition(a) {
      const res = [];
      for (; a > _1n; a >>= _1n) {
        if ((a & _1n) === _0n)
          res.unshift(0);
        else if ((a & _3n) === _3n) {
          res.unshift(-1);
          a += _1n;
        } else
          res.unshift(1);
      }
      return res;
    }
    function aNonEmpty(arr) {
      if (!Array.isArray(arr) || arr.length === 0)
        throw new Error("expected non-empty array");
    }
    function createBlsPairing(fields, G1, G2, params) {
      const { Fp2, Fp12 } = fields;
      const { twistType, ateLoopSize, xNegative, postPrecompute } = params;
      let lineFunction;
      if (twistType === "multiplicative") {
        lineFunction = (c0, c1, c2, f, Px, Py) => Fp12.mul014(f, c0, Fp2.mul(c1, Px), Fp2.mul(c2, Py));
      } else if (twistType === "divisive") {
        lineFunction = (c0, c1, c2, f, Px, Py) => Fp12.mul034(f, Fp2.mul(c2, Py), Fp2.mul(c1, Px), c0);
      } else
        throw new Error("bls: unknown twist type");
      const Fp2div2 = Fp2.div(Fp2.ONE, Fp2.mul(Fp2.ONE, _2n));
      function pointDouble(ell, Rx, Ry, Rz) {
        const t0 = Fp2.sqr(Ry);
        const t1 = Fp2.sqr(Rz);
        const t2 = Fp2.mulByB(Fp2.mul(t1, _3n));
        const t3 = Fp2.mul(t2, _3n);
        const t4 = Fp2.sub(Fp2.sub(Fp2.sqr(Fp2.add(Ry, Rz)), t1), t0);
        const c0 = Fp2.sub(t2, t0);
        const c1 = Fp2.mul(Fp2.sqr(Rx), _3n);
        const c2 = Fp2.neg(t4);
        ell.push([c0, c1, c2]);
        Rx = Fp2.mul(Fp2.mul(Fp2.mul(Fp2.sub(t0, t3), Rx), Ry), Fp2div2);
        Ry = Fp2.sub(Fp2.sqr(Fp2.mul(Fp2.add(t0, t3), Fp2div2)), Fp2.mul(Fp2.sqr(t2), _3n));
        Rz = Fp2.mul(t0, t4);
        return { Rx, Ry, Rz };
      }
      function pointAdd(ell, Rx, Ry, Rz, Qx, Qy) {
        const t0 = Fp2.sub(Ry, Fp2.mul(Qy, Rz));
        const t1 = Fp2.sub(Rx, Fp2.mul(Qx, Rz));
        const c0 = Fp2.sub(Fp2.mul(t0, Qx), Fp2.mul(t1, Qy));
        const c1 = Fp2.neg(t0);
        const c2 = t1;
        ell.push([c0, c1, c2]);
        const t2 = Fp2.sqr(t1);
        const t3 = Fp2.mul(t2, t1);
        const t4 = Fp2.mul(t2, Rx);
        const t5 = Fp2.add(Fp2.sub(t3, Fp2.mul(t4, _2n)), Fp2.mul(Fp2.sqr(t0), Rz));
        Rx = Fp2.mul(t1, t5);
        Ry = Fp2.sub(Fp2.mul(Fp2.sub(t4, t5), t0), Fp2.mul(t3, Ry));
        Rz = Fp2.mul(Rz, t3);
        return { Rx, Ry, Rz };
      }
      const ATE_NAF = NAfDecomposition(ateLoopSize);
      const calcPairingPrecomputes = (0, utils_ts_1.memoized)((point) => {
        const p = point;
        const { x, y } = p.toAffine();
        const Qx = x, Qy = y, negQy = Fp2.neg(y);
        let Rx = Qx, Ry = Qy, Rz = Fp2.ONE;
        const ell = [];
        for (const bit of ATE_NAF) {
          const cur = [];
          ({ Rx, Ry, Rz } = pointDouble(cur, Rx, Ry, Rz));
          if (bit)
            ({ Rx, Ry, Rz } = pointAdd(cur, Rx, Ry, Rz, Qx, bit === -1 ? negQy : Qy));
          ell.push(cur);
        }
        if (postPrecompute) {
          const last = ell[ell.length - 1];
          postPrecompute(Rx, Ry, Rz, Qx, Qy, pointAdd.bind(null, last));
        }
        return ell;
      });
      function millerLoopBatch(pairs, withFinalExponent = false) {
        let f12 = Fp12.ONE;
        if (pairs.length) {
          const ellLen = pairs[0][0].length;
          for (let i = 0; i < ellLen; i++) {
            f12 = Fp12.sqr(f12);
            for (const [ell, Px, Py] of pairs) {
              for (const [c0, c1, c2] of ell[i])
                f12 = lineFunction(c0, c1, c2, f12, Px, Py);
            }
          }
        }
        if (xNegative)
          f12 = Fp12.conjugate(f12);
        return withFinalExponent ? Fp12.finalExponentiate(f12) : f12;
      }
      function pairingBatch(pairs, withFinalExponent = true) {
        const res = [];
        (0, curve_ts_1.normalizeZ)(G1, pairs.map(({ g1 }) => g1));
        (0, curve_ts_1.normalizeZ)(G2, pairs.map(({ g2 }) => g2));
        for (const { g1, g2 } of pairs) {
          if (g1.is0() || g2.is0())
            throw new Error("pairing is not available for ZERO point");
          g1.assertValidity();
          g2.assertValidity();
          const Qa = g1.toAffine();
          res.push([calcPairingPrecomputes(g2), Qa.x, Qa.y]);
        }
        return millerLoopBatch(res, withFinalExponent);
      }
      function pairing(Q, P, withFinalExponent = true) {
        return pairingBatch([{ g1: Q, g2: P }], withFinalExponent);
      }
      return {
        Fp12,
        // NOTE: we re-export Fp12 here because pairing results are Fp12!
        millerLoopBatch,
        pairing,
        pairingBatch,
        calcPairingPrecomputes
      };
    }
    function createBlsSig(blsPairing, PubCurve, SigCurve, SignatureCoder, isSigG1) {
      const { Fp12, pairingBatch } = blsPairing;
      function normPub(point) {
        return point instanceof PubCurve.Point ? point : PubCurve.Point.fromHex(point);
      }
      function normSig(point) {
        return point instanceof SigCurve.Point ? point : SigCurve.Point.fromHex(point);
      }
      function amsg(m) {
        if (!(m instanceof SigCurve.Point))
          throw new Error(`expected valid message hashed to ${!isSigG1 ? "G2" : "G1"} curve`);
        return m;
      }
      const pair = !isSigG1 ? (a, b) => ({ g1: a, g2: b }) : (a, b) => ({ g1: b, g2: a });
      return {
        // P = pk x G
        getPublicKey(secretKey) {
          const sec = (0, weierstrass_ts_1._normFnElement)(PubCurve.Point.Fn, secretKey);
          return PubCurve.Point.BASE.multiply(sec);
        },
        // S = pk x H(m)
        sign(message2, secretKey, unusedArg) {
          if (unusedArg != null)
            throw new Error("sign() expects 2 arguments");
          const sec = (0, weierstrass_ts_1._normFnElement)(PubCurve.Point.Fn, secretKey);
          amsg(message2).assertValidity();
          return message2.multiply(sec);
        },
        // Checks if pairing of public key & hash is equal to pairing of generator & signature.
        // e(P, H(m)) == e(G, S)
        // e(S, G) == e(H(m), P)
        verify(signature2, message2, publicKey, unusedArg) {
          if (unusedArg != null)
            throw new Error("verify() expects 3 arguments");
          signature2 = normSig(signature2);
          publicKey = normPub(publicKey);
          const P = publicKey.negate();
          const G = PubCurve.Point.BASE;
          const Hm = amsg(message2);
          const S = signature2;
          const exp = pairingBatch([pair(P, Hm), pair(G, S)]);
          return Fp12.eql(exp, Fp12.ONE);
        },
        // https://ethresear.ch/t/fast-verification-of-multiple-bls-signatures/5407
        // e(G, S) = e(G, SUM(n)(Si)) = MUL(n)(e(G, Si))
        // TODO: maybe `{message: G2Hex, publicKey: G1Hex}[]` instead?
        verifyBatch(signature2, messages, publicKeys) {
          aNonEmpty(messages);
          if (publicKeys.length !== messages.length)
            throw new Error("amount of public keys and messages should be equal");
          const sig = normSig(signature2);
          const nMessages = messages;
          const nPublicKeys = publicKeys.map(normPub);
          const messagePubKeyMap = /* @__PURE__ */ new Map();
          for (let i = 0; i < nPublicKeys.length; i++) {
            const pub = nPublicKeys[i];
            const msg = nMessages[i];
            let keys = messagePubKeyMap.get(msg);
            if (keys === void 0) {
              keys = [];
              messagePubKeyMap.set(msg, keys);
            }
            keys.push(pub);
          }
          const paired = [];
          const G = PubCurve.Point.BASE;
          try {
            for (const [msg, keys] of messagePubKeyMap) {
              const groupPublicKey = keys.reduce((acc, msg2) => acc.add(msg2));
              paired.push(pair(groupPublicKey, msg));
            }
            paired.push(pair(G.negate(), sig));
            return Fp12.eql(pairingBatch(paired), Fp12.ONE);
          } catch {
            return false;
          }
        },
        // Adds a bunch of public key points together.
        // pk1 + pk2 + pk3 = pkA
        aggregatePublicKeys(publicKeys) {
          aNonEmpty(publicKeys);
          publicKeys = publicKeys.map((pub) => normPub(pub));
          const agg = publicKeys.reduce((sum, p) => sum.add(p), PubCurve.Point.ZERO);
          agg.assertValidity();
          return agg;
        },
        // Adds a bunch of signature points together.
        // pk1 + pk2 + pk3 = pkA
        aggregateSignatures(signatures) {
          aNonEmpty(signatures);
          signatures = signatures.map((sig) => normSig(sig));
          const agg = signatures.reduce((sum, s) => sum.add(s), SigCurve.Point.ZERO);
          agg.assertValidity();
          return agg;
        },
        hash(messageBytes, DST) {
          (0, utils_ts_1.abytes)(messageBytes);
          const opts = DST ? { DST } : void 0;
          return SigCurve.hashToCurve(messageBytes, opts);
        },
        Signature: SignatureCoder
      };
    }
    function bls2(CURVE) {
      const { Fp, Fr, Fp2, Fp6, Fp12 } = CURVE.fields;
      const G1_ = (0, weierstrass_ts_1.weierstrassPoints)(CURVE.G1);
      const G1 = Object.assign(G1_, (0, hash_to_curve_ts_1.createHasher)(G1_.Point, CURVE.G1.mapToCurve, {
        ...CURVE.htfDefaults,
        ...CURVE.G1.htfDefaults
      }));
      const G2_ = (0, weierstrass_ts_1.weierstrassPoints)(CURVE.G2);
      const G2 = Object.assign(G2_, (0, hash_to_curve_ts_1.createHasher)(G2_.Point, CURVE.G2.mapToCurve, {
        ...CURVE.htfDefaults,
        ...CURVE.G2.htfDefaults
      }));
      const pairingRes = createBlsPairing(CURVE.fields, G1.Point, G2.Point, {
        ...CURVE.params,
        postPrecompute: CURVE.postPrecompute
      });
      const { millerLoopBatch, pairing, pairingBatch, calcPairingPrecomputes } = pairingRes;
      const longSignatures = createBlsSig(pairingRes, G1, G2, CURVE.G2.Signature, false);
      const shortSignatures = createBlsSig(pairingRes, G2, G1, CURVE.G1.ShortSignature, true);
      const rand = CURVE.randomBytes || utils_ts_1.randomBytes;
      const randomSecretKey = () => {
        const length = (0, modular_ts_1.getMinHashLength)(Fr.ORDER);
        return (0, modular_ts_1.mapHashToField)(rand(length), Fr.ORDER);
      };
      const utils = {
        randomSecretKey,
        randomPrivateKey: randomSecretKey,
        calcPairingPrecomputes
      };
      const { ShortSignature } = CURVE.G1;
      const { Signature } = CURVE.G2;
      function normP1Hash(point, htfOpts) {
        return point instanceof G1.Point ? point : shortSignatures.hash((0, utils_ts_1.ensureBytes)("point", point), htfOpts?.DST);
      }
      function normP2Hash(point, htfOpts) {
        return point instanceof G2.Point ? point : longSignatures.hash((0, utils_ts_1.ensureBytes)("point", point), htfOpts?.DST);
      }
      function getPublicKey(privateKey) {
        return longSignatures.getPublicKey(privateKey).toBytes(true);
      }
      function getPublicKeyForShortSignatures(privateKey) {
        return shortSignatures.getPublicKey(privateKey).toBytes(true);
      }
      function sign(message2, privateKey, htfOpts) {
        const Hm = normP2Hash(message2, htfOpts);
        const S = longSignatures.sign(Hm, privateKey);
        return message2 instanceof G2.Point ? S : Signature.toBytes(S);
      }
      function signShortSignature(message2, privateKey, htfOpts) {
        const Hm = normP1Hash(message2, htfOpts);
        const S = shortSignatures.sign(Hm, privateKey);
        return message2 instanceof G1.Point ? S : ShortSignature.toBytes(S);
      }
      function verify(signature2, message2, publicKey, htfOpts) {
        const Hm = normP2Hash(message2, htfOpts);
        return longSignatures.verify(signature2, Hm, publicKey);
      }
      function verifyShortSignature(signature2, message2, publicKey, htfOpts) {
        const Hm = normP1Hash(message2, htfOpts);
        return shortSignatures.verify(signature2, Hm, publicKey);
      }
      function aggregatePublicKeys(publicKeys) {
        const agg = longSignatures.aggregatePublicKeys(publicKeys);
        return publicKeys[0] instanceof G1.Point ? agg : agg.toBytes(true);
      }
      function aggregateSignatures(signatures) {
        const agg = longSignatures.aggregateSignatures(signatures);
        return signatures[0] instanceof G2.Point ? agg : Signature.toBytes(agg);
      }
      function aggregateShortSignatures(signatures) {
        const agg = shortSignatures.aggregateSignatures(signatures);
        return signatures[0] instanceof G1.Point ? agg : ShortSignature.toBytes(agg);
      }
      function verifyBatch(signature2, messages, publicKeys, htfOpts) {
        const Hm = messages.map((m) => normP2Hash(m, htfOpts));
        return longSignatures.verifyBatch(signature2, Hm, publicKeys);
      }
      G1.Point.BASE.precompute(4);
      return {
        longSignatures,
        shortSignatures,
        millerLoopBatch,
        pairing,
        pairingBatch,
        verifyBatch,
        fields: {
          Fr,
          Fp,
          Fp2,
          Fp6,
          Fp12
        },
        params: {
          ateLoopSize: CURVE.params.ateLoopSize,
          twistType: CURVE.params.twistType,
          // deprecated
          r: CURVE.params.r,
          G1b: CURVE.G1.b,
          G2b: CURVE.G2.b
        },
        utils,
        // deprecated
        getPublicKey,
        getPublicKeyForShortSignatures,
        sign,
        signShortSignature,
        verify,
        verifyShortSignature,
        aggregatePublicKeys,
        aggregateSignatures,
        aggregateShortSignatures,
        G1,
        G2,
        Signature,
        ShortSignature
      };
    }
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/tower.js
var require_tower = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/abstract/tower.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.psiFrobenius = psiFrobenius;
    exports.tower12 = tower12;
    var utils_ts_1 = require_utils2();
    var mod = require_modular();
    var _0n = BigInt(0);
    var _1n = BigInt(1);
    var _2n = BigInt(2);
    var _3n = BigInt(3);
    function calcFrobeniusCoefficients(Fp, nonResidue, modulus, degree, num = 1, divisor) {
      const _divisor = BigInt(divisor === void 0 ? degree : divisor);
      const towerModulus = modulus ** BigInt(degree);
      const res = [];
      for (let i = 0; i < num; i++) {
        const a = BigInt(i + 1);
        const powers = [];
        for (let j = 0, qPower = _1n; j < degree; j++) {
          const power = (a * qPower - a) / _divisor % towerModulus;
          powers.push(Fp.pow(nonResidue, power));
          qPower *= modulus;
        }
        res.push(powers);
      }
      return res;
    }
    function psiFrobenius(Fp, Fp2, base) {
      const PSI_X = Fp2.pow(base, (Fp.ORDER - _1n) / _3n);
      const PSI_Y = Fp2.pow(base, (Fp.ORDER - _1n) / _2n);
      function psi(x, y) {
        const x2 = Fp2.mul(Fp2.frobeniusMap(x, 1), PSI_X);
        const y2 = Fp2.mul(Fp2.frobeniusMap(y, 1), PSI_Y);
        return [x2, y2];
      }
      const PSI2_X = Fp2.pow(base, (Fp.ORDER ** _2n - _1n) / _3n);
      const PSI2_Y = Fp2.pow(base, (Fp.ORDER ** _2n - _1n) / _2n);
      if (!Fp2.eql(PSI2_Y, Fp2.neg(Fp2.ONE)))
        throw new Error("psiFrobenius: PSI2_Y!==-1");
      function psi2(x, y) {
        return [Fp2.mul(x, PSI2_X), Fp2.neg(y)];
      }
      const mapAffine = (fn) => (c, P) => {
        const affine = P.toAffine();
        const p = fn(affine.x, affine.y);
        return c.fromAffine({ x: p[0], y: p[1] });
      };
      const G2psi = mapAffine(psi);
      const G2psi2 = mapAffine(psi2);
      return { psi, psi2, G2psi, G2psi2, PSI_X, PSI_Y, PSI2_X, PSI2_Y };
    }
    var Fp2fromBigTuple = (Fp, tuple) => {
      if (tuple.length !== 2)
        throw new Error("invalid tuple");
      const fps = tuple.map((n) => Fp.create(n));
      return { c0: fps[0], c1: fps[1] };
    };
    var _Field2 = class {
      constructor(Fp, opts = {}) {
        this.MASK = _1n;
        const ORDER = Fp.ORDER;
        const FP2_ORDER = ORDER * ORDER;
        this.Fp = Fp;
        this.ORDER = FP2_ORDER;
        this.BITS = (0, utils_ts_1.bitLen)(FP2_ORDER);
        this.BYTES = Math.ceil((0, utils_ts_1.bitLen)(FP2_ORDER) / 8);
        this.isLE = Fp.isLE;
        this.ZERO = { c0: Fp.ZERO, c1: Fp.ZERO };
        this.ONE = { c0: Fp.ONE, c1: Fp.ZERO };
        this.Fp_NONRESIDUE = Fp.create(opts.NONRESIDUE || BigInt(-1));
        this.Fp_div2 = Fp.div(Fp.ONE, _2n);
        this.NONRESIDUE = Fp2fromBigTuple(Fp, opts.FP2_NONRESIDUE);
        this.FROBENIUS_COEFFICIENTS = calcFrobeniusCoefficients(Fp, this.Fp_NONRESIDUE, Fp.ORDER, 2)[0];
        this.mulByB = opts.Fp2mulByB;
        Object.seal(this);
      }
      fromBigTuple(tuple) {
        return Fp2fromBigTuple(this.Fp, tuple);
      }
      create(num) {
        return num;
      }
      isValid({ c0, c1 }) {
        function isValidC(num, ORDER) {
          return typeof num === "bigint" && _0n <= num && num < ORDER;
        }
        return isValidC(c0, this.ORDER) && isValidC(c1, this.ORDER);
      }
      is0({ c0, c1 }) {
        return this.Fp.is0(c0) && this.Fp.is0(c1);
      }
      isValidNot0(num) {
        return !this.is0(num) && this.isValid(num);
      }
      eql({ c0, c1 }, { c0: r0, c1: r1 }) {
        return this.Fp.eql(c0, r0) && this.Fp.eql(c1, r1);
      }
      neg({ c0, c1 }) {
        return { c0: this.Fp.neg(c0), c1: this.Fp.neg(c1) };
      }
      pow(num, power) {
        return mod.FpPow(this, num, power);
      }
      invertBatch(nums) {
        return mod.FpInvertBatch(this, nums);
      }
      // Normalized
      add(f1, f2) {
        const { c0, c1 } = f1;
        const { c0: r0, c1: r1 } = f2;
        return {
          c0: this.Fp.add(c0, r0),
          c1: this.Fp.add(c1, r1)
        };
      }
      sub({ c0, c1 }, { c0: r0, c1: r1 }) {
        return {
          c0: this.Fp.sub(c0, r0),
          c1: this.Fp.sub(c1, r1)
        };
      }
      mul({ c0, c1 }, rhs) {
        const { Fp } = this;
        if (typeof rhs === "bigint")
          return { c0: Fp.mul(c0, rhs), c1: Fp.mul(c1, rhs) };
        const { c0: r0, c1: r1 } = rhs;
        let t1 = Fp.mul(c0, r0);
        let t2 = Fp.mul(c1, r1);
        const o0 = Fp.sub(t1, t2);
        const o1 = Fp.sub(Fp.mul(Fp.add(c0, c1), Fp.add(r0, r1)), Fp.add(t1, t2));
        return { c0: o0, c1: o1 };
      }
      sqr({ c0, c1 }) {
        const { Fp } = this;
        const a = Fp.add(c0, c1);
        const b = Fp.sub(c0, c1);
        const c = Fp.add(c0, c0);
        return { c0: Fp.mul(a, b), c1: Fp.mul(c, c1) };
      }
      // NonNormalized stuff
      addN(a, b) {
        return this.add(a, b);
      }
      subN(a, b) {
        return this.sub(a, b);
      }
      mulN(a, b) {
        return this.mul(a, b);
      }
      sqrN(a) {
        return this.sqr(a);
      }
      // Why inversion for bigint inside Fp instead of Fp2? it is even used in that context?
      div(lhs, rhs) {
        const { Fp } = this;
        return this.mul(lhs, typeof rhs === "bigint" ? Fp.inv(Fp.create(rhs)) : this.inv(rhs));
      }
      inv({ c0: a, c1: b }) {
        const { Fp } = this;
        const factor = Fp.inv(Fp.create(a * a + b * b));
        return { c0: Fp.mul(factor, Fp.create(a)), c1: Fp.mul(factor, Fp.create(-b)) };
      }
      sqrt(num) {
        const { Fp } = this;
        const Fp2 = this;
        const { c0, c1 } = num;
        if (Fp.is0(c1)) {
          if (mod.FpLegendre(Fp, c0) === 1)
            return Fp2.create({ c0: Fp.sqrt(c0), c1: Fp.ZERO });
          else
            return Fp2.create({ c0: Fp.ZERO, c1: Fp.sqrt(Fp.div(c0, this.Fp_NONRESIDUE)) });
        }
        const a = Fp.sqrt(Fp.sub(Fp.sqr(c0), Fp.mul(Fp.sqr(c1), this.Fp_NONRESIDUE)));
        let d = Fp.mul(Fp.add(a, c0), this.Fp_div2);
        const legendre = mod.FpLegendre(Fp, d);
        if (legendre === -1)
          d = Fp.sub(d, a);
        const a0 = Fp.sqrt(d);
        const candidateSqrt = Fp2.create({ c0: a0, c1: Fp.div(Fp.mul(c1, this.Fp_div2), a0) });
        if (!Fp2.eql(Fp2.sqr(candidateSqrt), num))
          throw new Error("Cannot find square root");
        const x1 = candidateSqrt;
        const x2 = Fp2.neg(x1);
        const { re: re1, im: im1 } = Fp2.reim(x1);
        const { re: re2, im: im2 } = Fp2.reim(x2);
        if (im1 > im2 || im1 === im2 && re1 > re2)
          return x1;
        return x2;
      }
      // Same as sgn0_m_eq_2 in RFC 9380
      isOdd(x) {
        const { re: x0, im: x1 } = this.reim(x);
        const sign_0 = x0 % _2n;
        const zero_0 = x0 === _0n;
        const sign_1 = x1 % _2n;
        return BigInt(sign_0 || zero_0 && sign_1) == _1n;
      }
      // Bytes util
      fromBytes(b) {
        const { Fp } = this;
        if (b.length !== this.BYTES)
          throw new Error("fromBytes invalid length=" + b.length);
        return { c0: Fp.fromBytes(b.subarray(0, Fp.BYTES)), c1: Fp.fromBytes(b.subarray(Fp.BYTES)) };
      }
      toBytes({ c0, c1 }) {
        return (0, utils_ts_1.concatBytes)(this.Fp.toBytes(c0), this.Fp.toBytes(c1));
      }
      cmov({ c0, c1 }, { c0: r0, c1: r1 }, c) {
        return {
          c0: this.Fp.cmov(c0, r0, c),
          c1: this.Fp.cmov(c1, r1, c)
        };
      }
      reim({ c0, c1 }) {
        return { re: c0, im: c1 };
      }
      Fp4Square(a, b) {
        const Fp2 = this;
        const a2 = Fp2.sqr(a);
        const b2 = Fp2.sqr(b);
        return {
          first: Fp2.add(Fp2.mulByNonresidue(b2), a2),
          // b² * Nonresidue + a²
          second: Fp2.sub(Fp2.sub(Fp2.sqr(Fp2.add(a, b)), a2), b2)
          // (a + b)² - a² - b²
        };
      }
      // multiply by u + 1
      mulByNonresidue({ c0, c1 }) {
        return this.mul({ c0, c1 }, this.NONRESIDUE);
      }
      frobeniusMap({ c0, c1 }, power) {
        return {
          c0,
          c1: this.Fp.mul(c1, this.FROBENIUS_COEFFICIENTS[power % 2])
        };
      }
    };
    var _Field6 = class {
      constructor(Fp2) {
        this.MASK = _1n;
        this.Fp2 = Fp2;
        this.ORDER = Fp2.ORDER;
        this.BITS = 3 * Fp2.BITS;
        this.BYTES = 3 * Fp2.BYTES;
        this.isLE = Fp2.isLE;
        this.ZERO = { c0: Fp2.ZERO, c1: Fp2.ZERO, c2: Fp2.ZERO };
        this.ONE = { c0: Fp2.ONE, c1: Fp2.ZERO, c2: Fp2.ZERO };
        const { Fp } = Fp2;
        const frob = calcFrobeniusCoefficients(Fp2, Fp2.NONRESIDUE, Fp.ORDER, 6, 2, 3);
        this.FROBENIUS_COEFFICIENTS_1 = frob[0];
        this.FROBENIUS_COEFFICIENTS_2 = frob[1];
        Object.seal(this);
      }
      add({ c0, c1, c2 }, { c0: r0, c1: r1, c2: r2 }) {
        const { Fp2 } = this;
        return {
          c0: Fp2.add(c0, r0),
          c1: Fp2.add(c1, r1),
          c2: Fp2.add(c2, r2)
        };
      }
      sub({ c0, c1, c2 }, { c0: r0, c1: r1, c2: r2 }) {
        const { Fp2 } = this;
        return {
          c0: Fp2.sub(c0, r0),
          c1: Fp2.sub(c1, r1),
          c2: Fp2.sub(c2, r2)
        };
      }
      mul({ c0, c1, c2 }, rhs) {
        const { Fp2 } = this;
        if (typeof rhs === "bigint") {
          return {
            c0: Fp2.mul(c0, rhs),
            c1: Fp2.mul(c1, rhs),
            c2: Fp2.mul(c2, rhs)
          };
        }
        const { c0: r0, c1: r1, c2: r2 } = rhs;
        const t0 = Fp2.mul(c0, r0);
        const t1 = Fp2.mul(c1, r1);
        const t2 = Fp2.mul(c2, r2);
        return {
          // t0 + (c1 + c2) * (r1 * r2) - (T1 + T2) * (u + 1)
          c0: Fp2.add(t0, Fp2.mulByNonresidue(Fp2.sub(Fp2.mul(Fp2.add(c1, c2), Fp2.add(r1, r2)), Fp2.add(t1, t2)))),
          // (c0 + c1) * (r0 + r1) - (T0 + T1) + T2 * (u + 1)
          c1: Fp2.add(Fp2.sub(Fp2.mul(Fp2.add(c0, c1), Fp2.add(r0, r1)), Fp2.add(t0, t1)), Fp2.mulByNonresidue(t2)),
          // T1 + (c0 + c2) * (r0 + r2) - T0 + T2
          c2: Fp2.sub(Fp2.add(t1, Fp2.mul(Fp2.add(c0, c2), Fp2.add(r0, r2))), Fp2.add(t0, t2))
        };
      }
      sqr({ c0, c1, c2 }) {
        const { Fp2 } = this;
        let t0 = Fp2.sqr(c0);
        let t1 = Fp2.mul(Fp2.mul(c0, c1), _2n);
        let t3 = Fp2.mul(Fp2.mul(c1, c2), _2n);
        let t4 = Fp2.sqr(c2);
        return {
          c0: Fp2.add(Fp2.mulByNonresidue(t3), t0),
          // T3 * (u + 1) + T0
          c1: Fp2.add(Fp2.mulByNonresidue(t4), t1),
          // T4 * (u + 1) + T1
          // T1 + (c0 - c1 + c2)² + T3 - T0 - T4
          c2: Fp2.sub(Fp2.sub(Fp2.add(Fp2.add(t1, Fp2.sqr(Fp2.add(Fp2.sub(c0, c1), c2))), t3), t0), t4)
        };
      }
      addN(a, b) {
        return this.add(a, b);
      }
      subN(a, b) {
        return this.sub(a, b);
      }
      mulN(a, b) {
        return this.mul(a, b);
      }
      sqrN(a) {
        return this.sqr(a);
      }
      create(num) {
        return num;
      }
      isValid({ c0, c1, c2 }) {
        const { Fp2 } = this;
        return Fp2.isValid(c0) && Fp2.isValid(c1) && Fp2.isValid(c2);
      }
      is0({ c0, c1, c2 }) {
        const { Fp2 } = this;
        return Fp2.is0(c0) && Fp2.is0(c1) && Fp2.is0(c2);
      }
      isValidNot0(num) {
        return !this.is0(num) && this.isValid(num);
      }
      neg({ c0, c1, c2 }) {
        const { Fp2 } = this;
        return { c0: Fp2.neg(c0), c1: Fp2.neg(c1), c2: Fp2.neg(c2) };
      }
      eql({ c0, c1, c2 }, { c0: r0, c1: r1, c2: r2 }) {
        const { Fp2 } = this;
        return Fp2.eql(c0, r0) && Fp2.eql(c1, r1) && Fp2.eql(c2, r2);
      }
      sqrt(_) {
        return (0, utils_ts_1.notImplemented)();
      }
      // Do we need division by bigint at all? Should be done via order:
      div(lhs, rhs) {
        const { Fp2 } = this;
        const { Fp } = Fp2;
        return this.mul(lhs, typeof rhs === "bigint" ? Fp.inv(Fp.create(rhs)) : this.inv(rhs));
      }
      pow(num, power) {
        return mod.FpPow(this, num, power);
      }
      invertBatch(nums) {
        return mod.FpInvertBatch(this, nums);
      }
      inv({ c0, c1, c2 }) {
        const { Fp2 } = this;
        let t0 = Fp2.sub(Fp2.sqr(c0), Fp2.mulByNonresidue(Fp2.mul(c2, c1)));
        let t1 = Fp2.sub(Fp2.mulByNonresidue(Fp2.sqr(c2)), Fp2.mul(c0, c1));
        let t2 = Fp2.sub(Fp2.sqr(c1), Fp2.mul(c0, c2));
        let t4 = Fp2.inv(Fp2.add(Fp2.mulByNonresidue(Fp2.add(Fp2.mul(c2, t1), Fp2.mul(c1, t2))), Fp2.mul(c0, t0)));
        return { c0: Fp2.mul(t4, t0), c1: Fp2.mul(t4, t1), c2: Fp2.mul(t4, t2) };
      }
      // Bytes utils
      fromBytes(b) {
        const { Fp2 } = this;
        if (b.length !== this.BYTES)
          throw new Error("fromBytes invalid length=" + b.length);
        const B2 = Fp2.BYTES;
        return {
          c0: Fp2.fromBytes(b.subarray(0, B2)),
          c1: Fp2.fromBytes(b.subarray(B2, B2 * 2)),
          c2: Fp2.fromBytes(b.subarray(2 * B2))
        };
      }
      toBytes({ c0, c1, c2 }) {
        const { Fp2 } = this;
        return (0, utils_ts_1.concatBytes)(Fp2.toBytes(c0), Fp2.toBytes(c1), Fp2.toBytes(c2));
      }
      cmov({ c0, c1, c2 }, { c0: r0, c1: r1, c2: r2 }, c) {
        const { Fp2 } = this;
        return {
          c0: Fp2.cmov(c0, r0, c),
          c1: Fp2.cmov(c1, r1, c),
          c2: Fp2.cmov(c2, r2, c)
        };
      }
      fromBigSix(t) {
        const { Fp2 } = this;
        if (!Array.isArray(t) || t.length !== 6)
          throw new Error("invalid Fp6 usage");
        return {
          c0: Fp2.fromBigTuple(t.slice(0, 2)),
          c1: Fp2.fromBigTuple(t.slice(2, 4)),
          c2: Fp2.fromBigTuple(t.slice(4, 6))
        };
      }
      frobeniusMap({ c0, c1, c2 }, power) {
        const { Fp2 } = this;
        return {
          c0: Fp2.frobeniusMap(c0, power),
          c1: Fp2.mul(Fp2.frobeniusMap(c1, power), this.FROBENIUS_COEFFICIENTS_1[power % 6]),
          c2: Fp2.mul(Fp2.frobeniusMap(c2, power), this.FROBENIUS_COEFFICIENTS_2[power % 6])
        };
      }
      mulByFp2({ c0, c1, c2 }, rhs) {
        const { Fp2 } = this;
        return {
          c0: Fp2.mul(c0, rhs),
          c1: Fp2.mul(c1, rhs),
          c2: Fp2.mul(c2, rhs)
        };
      }
      mulByNonresidue({ c0, c1, c2 }) {
        const { Fp2 } = this;
        return { c0: Fp2.mulByNonresidue(c2), c1: c0, c2: c1 };
      }
      // Sparse multiplication
      mul1({ c0, c1, c2 }, b1) {
        const { Fp2 } = this;
        return {
          c0: Fp2.mulByNonresidue(Fp2.mul(c2, b1)),
          c1: Fp2.mul(c0, b1),
          c2: Fp2.mul(c1, b1)
        };
      }
      // Sparse multiplication
      mul01({ c0, c1, c2 }, b0, b1) {
        const { Fp2 } = this;
        let t0 = Fp2.mul(c0, b0);
        let t1 = Fp2.mul(c1, b1);
        return {
          // ((c1 + c2) * b1 - T1) * (u + 1) + T0
          c0: Fp2.add(Fp2.mulByNonresidue(Fp2.sub(Fp2.mul(Fp2.add(c1, c2), b1), t1)), t0),
          // (b0 + b1) * (c0 + c1) - T0 - T1
          c1: Fp2.sub(Fp2.sub(Fp2.mul(Fp2.add(b0, b1), Fp2.add(c0, c1)), t0), t1),
          // (c0 + c2) * b0 - T0 + T1
          c2: Fp2.add(Fp2.sub(Fp2.mul(Fp2.add(c0, c2), b0), t0), t1)
        };
      }
    };
    var _Field12 = class {
      constructor(Fp6, opts) {
        this.MASK = _1n;
        const { Fp2 } = Fp6;
        const { Fp } = Fp2;
        this.Fp6 = Fp6;
        this.ORDER = Fp2.ORDER;
        this.BITS = 2 * Fp6.BITS;
        this.BYTES = 2 * Fp6.BYTES;
        this.isLE = Fp6.isLE;
        this.ZERO = { c0: Fp6.ZERO, c1: Fp6.ZERO };
        this.ONE = { c0: Fp6.ONE, c1: Fp6.ZERO };
        this.FROBENIUS_COEFFICIENTS = calcFrobeniusCoefficients(Fp2, Fp2.NONRESIDUE, Fp.ORDER, 12, 1, 6)[0];
        this.X_LEN = opts.X_LEN;
        this.finalExponentiate = opts.Fp12finalExponentiate;
      }
      create(num) {
        return num;
      }
      isValid({ c0, c1 }) {
        const { Fp6 } = this;
        return Fp6.isValid(c0) && Fp6.isValid(c1);
      }
      is0({ c0, c1 }) {
        const { Fp6 } = this;
        return Fp6.is0(c0) && Fp6.is0(c1);
      }
      isValidNot0(num) {
        return !this.is0(num) && this.isValid(num);
      }
      neg({ c0, c1 }) {
        const { Fp6 } = this;
        return { c0: Fp6.neg(c0), c1: Fp6.neg(c1) };
      }
      eql({ c0, c1 }, { c0: r0, c1: r1 }) {
        const { Fp6 } = this;
        return Fp6.eql(c0, r0) && Fp6.eql(c1, r1);
      }
      sqrt(_) {
        (0, utils_ts_1.notImplemented)();
      }
      inv({ c0, c1 }) {
        const { Fp6 } = this;
        let t = Fp6.inv(Fp6.sub(Fp6.sqr(c0), Fp6.mulByNonresidue(Fp6.sqr(c1))));
        return { c0: Fp6.mul(c0, t), c1: Fp6.neg(Fp6.mul(c1, t)) };
      }
      div(lhs, rhs) {
        const { Fp6 } = this;
        const { Fp2 } = Fp6;
        const { Fp } = Fp2;
        return this.mul(lhs, typeof rhs === "bigint" ? Fp.inv(Fp.create(rhs)) : this.inv(rhs));
      }
      pow(num, power) {
        return mod.FpPow(this, num, power);
      }
      invertBatch(nums) {
        return mod.FpInvertBatch(this, nums);
      }
      // Normalized
      add({ c0, c1 }, { c0: r0, c1: r1 }) {
        const { Fp6 } = this;
        return {
          c0: Fp6.add(c0, r0),
          c1: Fp6.add(c1, r1)
        };
      }
      sub({ c0, c1 }, { c0: r0, c1: r1 }) {
        const { Fp6 } = this;
        return {
          c0: Fp6.sub(c0, r0),
          c1: Fp6.sub(c1, r1)
        };
      }
      mul({ c0, c1 }, rhs) {
        const { Fp6 } = this;
        if (typeof rhs === "bigint")
          return { c0: Fp6.mul(c0, rhs), c1: Fp6.mul(c1, rhs) };
        let { c0: r0, c1: r1 } = rhs;
        let t1 = Fp6.mul(c0, r0);
        let t2 = Fp6.mul(c1, r1);
        return {
          c0: Fp6.add(t1, Fp6.mulByNonresidue(t2)),
          // T1 + T2 * v
          // (c0 + c1) * (r0 + r1) - (T1 + T2)
          c1: Fp6.sub(Fp6.mul(Fp6.add(c0, c1), Fp6.add(r0, r1)), Fp6.add(t1, t2))
        };
      }
      sqr({ c0, c1 }) {
        const { Fp6 } = this;
        let ab = Fp6.mul(c0, c1);
        return {
          // (c1 * v + c0) * (c0 + c1) - AB - AB * v
          c0: Fp6.sub(Fp6.sub(Fp6.mul(Fp6.add(Fp6.mulByNonresidue(c1), c0), Fp6.add(c0, c1)), ab), Fp6.mulByNonresidue(ab)),
          c1: Fp6.add(ab, ab)
        };
      }
      // NonNormalized stuff
      addN(a, b) {
        return this.add(a, b);
      }
      subN(a, b) {
        return this.sub(a, b);
      }
      mulN(a, b) {
        return this.mul(a, b);
      }
      sqrN(a) {
        return this.sqr(a);
      }
      // Bytes utils
      fromBytes(b) {
        const { Fp6 } = this;
        if (b.length !== this.BYTES)
          throw new Error("fromBytes invalid length=" + b.length);
        return {
          c0: Fp6.fromBytes(b.subarray(0, Fp6.BYTES)),
          c1: Fp6.fromBytes(b.subarray(Fp6.BYTES))
        };
      }
      toBytes({ c0, c1 }) {
        const { Fp6 } = this;
        return (0, utils_ts_1.concatBytes)(Fp6.toBytes(c0), Fp6.toBytes(c1));
      }
      cmov({ c0, c1 }, { c0: r0, c1: r1 }, c) {
        const { Fp6 } = this;
        return {
          c0: Fp6.cmov(c0, r0, c),
          c1: Fp6.cmov(c1, r1, c)
        };
      }
      // Utils
      // toString() {
      //   return '' + 'Fp12(' + this.c0 + this.c1 + '* w');
      // },
      // fromTuple(c: [Fp6, Fp6]) {
      //   return new Fp12(...c);
      // }
      fromBigTwelve(t) {
        const { Fp6 } = this;
        return {
          c0: Fp6.fromBigSix(t.slice(0, 6)),
          c1: Fp6.fromBigSix(t.slice(6, 12))
        };
      }
      // Raises to q**i -th power
      frobeniusMap(lhs, power) {
        const { Fp6 } = this;
        const { Fp2 } = Fp6;
        const { c0, c1, c2 } = Fp6.frobeniusMap(lhs.c1, power);
        const coeff = this.FROBENIUS_COEFFICIENTS[power % 12];
        return {
          c0: Fp6.frobeniusMap(lhs.c0, power),
          c1: Fp6.create({
            c0: Fp2.mul(c0, coeff),
            c1: Fp2.mul(c1, coeff),
            c2: Fp2.mul(c2, coeff)
          })
        };
      }
      mulByFp2({ c0, c1 }, rhs) {
        const { Fp6 } = this;
        return {
          c0: Fp6.mulByFp2(c0, rhs),
          c1: Fp6.mulByFp2(c1, rhs)
        };
      }
      conjugate({ c0, c1 }) {
        return { c0, c1: this.Fp6.neg(c1) };
      }
      // Sparse multiplication
      mul014({ c0, c1 }, o0, o1, o4) {
        const { Fp6 } = this;
        const { Fp2 } = Fp6;
        let t0 = Fp6.mul01(c0, o0, o1);
        let t1 = Fp6.mul1(c1, o4);
        return {
          c0: Fp6.add(Fp6.mulByNonresidue(t1), t0),
          // T1 * v + T0
          // (c1 + c0) * [o0, o1+o4] - T0 - T1
          c1: Fp6.sub(Fp6.sub(Fp6.mul01(Fp6.add(c1, c0), o0, Fp2.add(o1, o4)), t0), t1)
        };
      }
      mul034({ c0, c1 }, o0, o3, o4) {
        const { Fp6 } = this;
        const { Fp2 } = Fp6;
        const a = Fp6.create({
          c0: Fp2.mul(c0.c0, o0),
          c1: Fp2.mul(c0.c1, o0),
          c2: Fp2.mul(c0.c2, o0)
        });
        const b = Fp6.mul01(c1, o3, o4);
        const e = Fp6.mul01(Fp6.add(c0, c1), Fp2.add(o0, o3), o4);
        return {
          c0: Fp6.add(Fp6.mulByNonresidue(b), a),
          c1: Fp6.sub(e, Fp6.add(a, b))
        };
      }
      // A cyclotomic group is a subgroup of Fp^n defined by
      //   GΦₙ(p) = {α ∈ Fpⁿ : α^Φₙ(p) = 1}
      // The result of any pairing is in a cyclotomic subgroup
      // https://eprint.iacr.org/2009/565.pdf
      // https://eprint.iacr.org/2010/354.pdf
      _cyclotomicSquare({ c0, c1 }) {
        const { Fp6 } = this;
        const { Fp2 } = Fp6;
        const { c0: c0c0, c1: c0c1, c2: c0c2 } = c0;
        const { c0: c1c0, c1: c1c1, c2: c1c2 } = c1;
        const { first: t3, second: t4 } = Fp2.Fp4Square(c0c0, c1c1);
        const { first: t5, second: t6 } = Fp2.Fp4Square(c1c0, c0c2);
        const { first: t7, second: t8 } = Fp2.Fp4Square(c0c1, c1c2);
        const t9 = Fp2.mulByNonresidue(t8);
        return {
          c0: Fp6.create({
            c0: Fp2.add(Fp2.mul(Fp2.sub(t3, c0c0), _2n), t3),
            // 2 * (T3 - c0c0)  + T3
            c1: Fp2.add(Fp2.mul(Fp2.sub(t5, c0c1), _2n), t5),
            // 2 * (T5 - c0c1)  + T5
            c2: Fp2.add(Fp2.mul(Fp2.sub(t7, c0c2), _2n), t7)
          }),
          // 2 * (T7 - c0c2)  + T7
          c1: Fp6.create({
            c0: Fp2.add(Fp2.mul(Fp2.add(t9, c1c0), _2n), t9),
            // 2 * (T9 + c1c0) + T9
            c1: Fp2.add(Fp2.mul(Fp2.add(t4, c1c1), _2n), t4),
            // 2 * (T4 + c1c1) + T4
            c2: Fp2.add(Fp2.mul(Fp2.add(t6, c1c2), _2n), t6)
          })
        };
      }
      // https://eprint.iacr.org/2009/565.pdf
      _cyclotomicExp(num, n) {
        let z = this.ONE;
        for (let i = this.X_LEN - 1; i >= 0; i--) {
          z = this._cyclotomicSquare(z);
          if ((0, utils_ts_1.bitGet)(n, i))
            z = this.mul(z, num);
        }
        return z;
      }
    };
    function tower12(opts) {
      const Fp = mod.Field(opts.ORDER);
      const Fp2 = new _Field2(Fp, opts);
      const Fp6 = new _Field6(Fp2);
      const Fp12 = new _Field12(Fp6, opts);
      return { Fp, Fp2, Fp6, Fp12 };
    }
  }
});

// ../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/bls12-381.js
var require_bls12_381 = __commonJS({
  "../.cache/glm52-drand-client-1.4.2/node_modules/@noble/curves/bls12-381.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.bls12_381 = exports.bls12_381_Fr = void 0;
    var sha2_js_1 = require_sha2();
    var bls_ts_1 = require_bls();
    var modular_ts_1 = require_modular();
    var utils_ts_1 = require_utils2();
    var hash_to_curve_ts_1 = require_hash_to_curve();
    var tower_ts_1 = require_tower();
    var weierstrass_ts_1 = require_weierstrass();
    var _0n = BigInt(0);
    var _1n = BigInt(1);
    var _2n = BigInt(2);
    var _3n = BigInt(3);
    var _4n = BigInt(4);
    var BLS_X = BigInt("0xd201000000010000");
    var BLS_X_LEN = (0, utils_ts_1.bitLen)(BLS_X);
    var bls12_381_CURVE_G1 = {
      p: BigInt("0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"),
      n: BigInt("0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"),
      h: BigInt("0x396c8c005555e1568c00aaab0000aaab"),
      a: _0n,
      b: _4n,
      Gx: BigInt("0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"),
      Gy: BigInt("0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1")
    };
    exports.bls12_381_Fr = (0, modular_ts_1.Field)(bls12_381_CURVE_G1.n, {
      modFromBytes: true,
      isLE: true
    });
    var { Fp, Fp2, Fp6, Fp12 } = (0, tower_ts_1.tower12)({
      ORDER: bls12_381_CURVE_G1.p,
      X_LEN: BLS_X_LEN,
      // Finite extension field over irreducible polynominal.
      // Fp(u) / (u² - β) where β = -1
      FP2_NONRESIDUE: [_1n, _1n],
      Fp2mulByB: ({ c0, c1 }) => {
        const t0 = Fp.mul(c0, _4n);
        const t1 = Fp.mul(c1, _4n);
        return { c0: Fp.sub(t0, t1), c1: Fp.add(t0, t1) };
      },
      Fp12finalExponentiate: (num) => {
        const x = BLS_X;
        const t0 = Fp12.div(Fp12.frobeniusMap(num, 6), num);
        const t1 = Fp12.mul(Fp12.frobeniusMap(t0, 2), t0);
        const t2 = Fp12.conjugate(Fp12._cyclotomicExp(t1, x));
        const t3 = Fp12.mul(Fp12.conjugate(Fp12._cyclotomicSquare(t1)), t2);
        const t4 = Fp12.conjugate(Fp12._cyclotomicExp(t3, x));
        const t5 = Fp12.conjugate(Fp12._cyclotomicExp(t4, x));
        const t6 = Fp12.mul(Fp12.conjugate(Fp12._cyclotomicExp(t5, x)), Fp12._cyclotomicSquare(t2));
        const t7 = Fp12.conjugate(Fp12._cyclotomicExp(t6, x));
        const t2_t5_pow_q2 = Fp12.frobeniusMap(Fp12.mul(t2, t5), 2);
        const t4_t1_pow_q3 = Fp12.frobeniusMap(Fp12.mul(t4, t1), 3);
        const t6_t1c_pow_q1 = Fp12.frobeniusMap(Fp12.mul(t6, Fp12.conjugate(t1)), 1);
        const t7_t3c_t1 = Fp12.mul(Fp12.mul(t7, Fp12.conjugate(t3)), t1);
        return Fp12.mul(Fp12.mul(Fp12.mul(t2_t5_pow_q2, t4_t1_pow_q3), t6_t1c_pow_q1), t7_t3c_t1);
      }
    });
    var { G2psi, G2psi2 } = (0, tower_ts_1.psiFrobenius)(Fp, Fp2, Fp2.div(Fp2.ONE, Fp2.NONRESIDUE));
    var htfDefaults = Object.freeze({
      DST: "BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_",
      encodeDST: "BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_",
      p: Fp.ORDER,
      m: 2,
      k: 128,
      expand: "xmd",
      hash: sha2_js_1.sha256
    });
    var bls12_381_CURVE_G2 = {
      p: Fp2.ORDER,
      n: bls12_381_CURVE_G1.n,
      h: BigInt("0x5d543a95414e7f1091d50792876a202cd91de4547085abaa68a205b2e5a7ddfa628f1cb4d9e82ef21537e293a6691ae1616ec6e786f0c70cf1c38e31c7238e5"),
      a: Fp2.ZERO,
      b: Fp2.fromBigTuple([_4n, _4n]),
      Gx: Fp2.fromBigTuple([
        BigInt("0x024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8"),
        BigInt("0x13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e")
      ]),
      Gy: Fp2.fromBigTuple([
        BigInt("0x0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801"),
        BigInt("0x0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be")
      ])
    };
    var COMPZERO = setMask(Fp.toBytes(_0n), { infinity: true, compressed: true });
    function parseMask(bytes) {
      bytes = bytes.slice();
      const mask = bytes[0] & 224;
      const compressed = !!(mask >> 7 & 1);
      const infinity = !!(mask >> 6 & 1);
      const sort = !!(mask >> 5 & 1);
      bytes[0] &= 31;
      return { compressed, infinity, sort, value: bytes };
    }
    function setMask(bytes, mask) {
      if (bytes[0] & 224)
        throw new Error("setMask: non-empty mask");
      if (mask.compressed)
        bytes[0] |= 128;
      if (mask.infinity)
        bytes[0] |= 64;
      if (mask.sort)
        bytes[0] |= 32;
      return bytes;
    }
    function pointG1ToBytes(_c, point, isComp) {
      const { BYTES: L, ORDER: P } = Fp;
      const is0 = point.is0();
      const { x, y } = point.toAffine();
      if (isComp) {
        if (is0)
          return COMPZERO.slice();
        const sort = Boolean(y * _2n / P);
        return setMask((0, utils_ts_1.numberToBytesBE)(x, L), { compressed: true, sort });
      } else {
        if (is0) {
          return (0, utils_ts_1.concatBytes)(Uint8Array.of(64), new Uint8Array(2 * L - 1));
        } else {
          return (0, utils_ts_1.concatBytes)((0, utils_ts_1.numberToBytesBE)(x, L), (0, utils_ts_1.numberToBytesBE)(y, L));
        }
      }
    }
    function signatureG1ToBytes(point) {
      point.assertValidity();
      const { BYTES: L, ORDER: P } = Fp;
      const { x, y } = point.toAffine();
      if (point.is0())
        return COMPZERO.slice();
      const sort = Boolean(y * _2n / P);
      return setMask((0, utils_ts_1.numberToBytesBE)(x, L), { compressed: true, sort });
    }
    function pointG1FromBytes(bytes) {
      const { compressed, infinity, sort, value } = parseMask(bytes);
      const { BYTES: L, ORDER: P } = Fp;
      if (value.length === 48 && compressed) {
        const compressedValue = (0, utils_ts_1.bytesToNumberBE)(value);
        const x = Fp.create(compressedValue & (0, utils_ts_1.bitMask)(Fp.BITS));
        if (infinity) {
          if (x !== _0n)
            throw new Error("invalid G1 point: non-empty, at infinity, with compression");
          return { x: _0n, y: _0n };
        }
        const right = Fp.add(Fp.pow(x, _3n), Fp.create(bls12_381_CURVE_G1.b));
        let y = Fp.sqrt(right);
        if (!y)
          throw new Error("invalid G1 point: compressed point");
        if (y * _2n / P !== BigInt(sort))
          y = Fp.neg(y);
        return { x: Fp.create(x), y: Fp.create(y) };
      } else if (value.length === 96 && !compressed) {
        const x = (0, utils_ts_1.bytesToNumberBE)(value.subarray(0, L));
        const y = (0, utils_ts_1.bytesToNumberBE)(value.subarray(L));
        if (infinity) {
          if (x !== _0n || y !== _0n)
            throw new Error("G1: non-empty point at infinity");
          return exports.bls12_381.G1.Point.ZERO.toAffine();
        }
        return { x: Fp.create(x), y: Fp.create(y) };
      } else {
        throw new Error("invalid G1 point: expected 48/96 bytes");
      }
    }
    function signatureG1FromBytes(hex) {
      const { infinity, sort, value } = parseMask((0, utils_ts_1.ensureBytes)("signatureHex", hex, 48));
      const P = Fp.ORDER;
      const Point = exports.bls12_381.G1.Point;
      const compressedValue = (0, utils_ts_1.bytesToNumberBE)(value);
      if (infinity)
        return Point.ZERO;
      const x = Fp.create(compressedValue & (0, utils_ts_1.bitMask)(Fp.BITS));
      const right = Fp.add(Fp.pow(x, _3n), Fp.create(bls12_381_CURVE_G1.b));
      let y = Fp.sqrt(right);
      if (!y)
        throw new Error("invalid G1 point: compressed");
      const aflag = BigInt(sort);
      if (y * _2n / P !== aflag)
        y = Fp.neg(y);
      const point = Point.fromAffine({ x, y });
      point.assertValidity();
      return point;
    }
    function pointG2ToBytes(_c, point, isComp) {
      const { BYTES: L, ORDER: P } = Fp;
      const is0 = point.is0();
      const { x, y } = point.toAffine();
      if (isComp) {
        if (is0)
          return (0, utils_ts_1.concatBytes)(COMPZERO, (0, utils_ts_1.numberToBytesBE)(_0n, L));
        const flag = Boolean(y.c1 === _0n ? y.c0 * _2n / P : y.c1 * _2n / P);
        return (0, utils_ts_1.concatBytes)(setMask((0, utils_ts_1.numberToBytesBE)(x.c1, L), { compressed: true, sort: flag }), (0, utils_ts_1.numberToBytesBE)(x.c0, L));
      } else {
        if (is0)
          return (0, utils_ts_1.concatBytes)(Uint8Array.of(64), new Uint8Array(4 * L - 1));
        const { re: x0, im: x1 } = Fp2.reim(x);
        const { re: y0, im: y1 } = Fp2.reim(y);
        return (0, utils_ts_1.concatBytes)((0, utils_ts_1.numberToBytesBE)(x1, L), (0, utils_ts_1.numberToBytesBE)(x0, L), (0, utils_ts_1.numberToBytesBE)(y1, L), (0, utils_ts_1.numberToBytesBE)(y0, L));
      }
    }
    function signatureG2ToBytes(point) {
      point.assertValidity();
      const { BYTES: L } = Fp;
      if (point.is0())
        return (0, utils_ts_1.concatBytes)(COMPZERO, (0, utils_ts_1.numberToBytesBE)(_0n, L));
      const { x, y } = point.toAffine();
      const { re: x0, im: x1 } = Fp2.reim(x);
      const { re: y0, im: y1 } = Fp2.reim(y);
      const tmp = y1 > _0n ? y1 * _2n : y0 * _2n;
      const sort = Boolean(tmp / Fp.ORDER & _1n);
      const z2 = x0;
      return (0, utils_ts_1.concatBytes)(setMask((0, utils_ts_1.numberToBytesBE)(x1, L), { sort, compressed: true }), (0, utils_ts_1.numberToBytesBE)(z2, L));
    }
    function pointG2FromBytes(bytes) {
      const { BYTES: L, ORDER: P } = Fp;
      const { compressed, infinity, sort, value } = parseMask(bytes);
      if (!compressed && !infinity && sort || // 00100000
      !compressed && infinity && sort || // 01100000
      sort && infinity && compressed) {
        throw new Error("invalid encoding flag: " + (bytes[0] & 224));
      }
      const slc = (b, from, to) => (0, utils_ts_1.bytesToNumberBE)(b.slice(from, to));
      if (value.length === 96 && compressed) {
        if (infinity) {
          if (value.reduce((p, c) => p !== 0 ? c + 1 : c, 0) > 0) {
            throw new Error("invalid G2 point: compressed");
          }
          return { x: Fp2.ZERO, y: Fp2.ZERO };
        }
        const x_1 = slc(value, 0, L);
        const x_0 = slc(value, L, 2 * L);
        const x = Fp2.create({ c0: Fp.create(x_0), c1: Fp.create(x_1) });
        const right = Fp2.add(Fp2.pow(x, _3n), bls12_381_CURVE_G2.b);
        let y = Fp2.sqrt(right);
        const Y_bit = y.c1 === _0n ? y.c0 * _2n / P : y.c1 * _2n / P ? _1n : _0n;
        y = sort && Y_bit > 0 ? y : Fp2.neg(y);
        return { x, y };
      } else if (value.length === 192 && !compressed) {
        if (infinity) {
          if (value.reduce((p, c) => p !== 0 ? c + 1 : c, 0) > 0) {
            throw new Error("invalid G2 point: uncompressed");
          }
          return { x: Fp2.ZERO, y: Fp2.ZERO };
        }
        const x1 = slc(value, 0 * L, 1 * L);
        const x0 = slc(value, 1 * L, 2 * L);
        const y1 = slc(value, 2 * L, 3 * L);
        const y0 = slc(value, 3 * L, 4 * L);
        return { x: Fp2.fromBigTuple([x0, x1]), y: Fp2.fromBigTuple([y0, y1]) };
      } else {
        throw new Error("invalid G2 point: expected 96/192 bytes");
      }
    }
    function signatureG2FromBytes(hex) {
      const { ORDER: P } = Fp;
      const { infinity, sort, value } = parseMask((0, utils_ts_1.ensureBytes)("signatureHex", hex));
      const Point = exports.bls12_381.G2.Point;
      const half = value.length / 2;
      if (half !== 48 && half !== 96)
        throw new Error("invalid compressed signature length, expected 96/192 bytes");
      const z1 = (0, utils_ts_1.bytesToNumberBE)(value.slice(0, half));
      const z2 = (0, utils_ts_1.bytesToNumberBE)(value.slice(half));
      if (infinity)
        return Point.ZERO;
      const x1 = Fp.create(z1 & (0, utils_ts_1.bitMask)(Fp.BITS));
      const x2 = Fp.create(z2);
      const x = Fp2.create({ c0: x2, c1: x1 });
      const y2 = Fp2.add(Fp2.pow(x, _3n), bls12_381_CURVE_G2.b);
      let y = Fp2.sqrt(y2);
      if (!y)
        throw new Error("Failed to find a square root");
      const { re: y0, im: y1 } = Fp2.reim(y);
      const aflag1 = BigInt(sort);
      const isGreater = y1 > _0n && y1 * _2n / P !== aflag1;
      const is0 = y1 === _0n && y0 * _2n / P !== aflag1;
      if (isGreater || is0)
        y = Fp2.neg(y);
      const point = Point.fromAffine({ x, y });
      point.assertValidity();
      return point;
    }
    exports.bls12_381 = (0, bls_ts_1.bls)({
      // Fields
      fields: {
        Fp,
        Fp2,
        Fp6,
        Fp12,
        Fr: exports.bls12_381_Fr
      },
      // G1: y² = x³ + 4
      G1: {
        ...bls12_381_CURVE_G1,
        Fp,
        htfDefaults: { ...htfDefaults, m: 1, DST: "BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_" },
        wrapPrivateKey: true,
        allowInfinityPoint: true,
        // Checks is the point resides in prime-order subgroup.
        // point.isTorsionFree() should return true for valid points
        // It returns false for shitty points.
        // https://eprint.iacr.org/2021/1130.pdf
        isTorsionFree: (c, point) => {
          const beta = BigInt("0x5f19672fdf76ce51ba69c6076a0f77eaddb3a93be6f89688de17d813620a00022e01fffffffefffe");
          const phi = new c(Fp.mul(point.X, beta), point.Y, point.Z);
          const xP = point.multiplyUnsafe(BLS_X).negate();
          const u2P = xP.multiplyUnsafe(BLS_X);
          return u2P.equals(phi);
        },
        // Clear cofactor of G1
        // https://eprint.iacr.org/2019/403
        clearCofactor: (_c, point) => {
          return point.multiplyUnsafe(BLS_X).add(point);
        },
        mapToCurve: mapToG1,
        fromBytes: pointG1FromBytes,
        toBytes: pointG1ToBytes,
        ShortSignature: {
          fromBytes(bytes) {
            (0, utils_ts_1.abytes)(bytes);
            return signatureG1FromBytes(bytes);
          },
          fromHex(hex) {
            return signatureG1FromBytes(hex);
          },
          toBytes(point) {
            return signatureG1ToBytes(point);
          },
          toRawBytes(point) {
            return signatureG1ToBytes(point);
          },
          toHex(point) {
            return (0, utils_ts_1.bytesToHex)(signatureG1ToBytes(point));
          }
        }
      },
      G2: {
        ...bls12_381_CURVE_G2,
        Fp: Fp2,
        // https://datatracker.ietf.org/doc/html/rfc9380#name-clearing-the-cofactor
        // https://datatracker.ietf.org/doc/html/rfc9380#name-cofactor-clearing-for-bls12
        hEff: BigInt("0xbc69f08f2ee75b3584c6a0ea91b352888e2a8e9145ad7689986ff031508ffe1329c2f178731db956d82bf015d1212b02ec0ec69d7477c1ae954cbc06689f6a359894c0adebbf6b4e8020005aaa95551"),
        htfDefaults: { ...htfDefaults },
        wrapPrivateKey: true,
        allowInfinityPoint: true,
        mapToCurve: mapToG2,
        // Checks is the point resides in prime-order subgroup.
        // point.isTorsionFree() should return true for valid points
        // It returns false for shitty points.
        // https://eprint.iacr.org/2021/1130.pdf
        // Older version: https://eprint.iacr.org/2019/814.pdf
        isTorsionFree: (c, P) => {
          return P.multiplyUnsafe(BLS_X).negate().equals(G2psi(c, P));
        },
        // Maps the point into the prime-order subgroup G2.
        // clear_cofactor_bls12381_g2 from RFC 9380.
        // https://eprint.iacr.org/2017/419.pdf
        // prettier-ignore
        clearCofactor: (c, P) => {
          const x = BLS_X;
          let t1 = P.multiplyUnsafe(x).negate();
          let t2 = G2psi(c, P);
          let t3 = P.double();
          t3 = G2psi2(c, t3);
          t3 = t3.subtract(t2);
          t2 = t1.add(t2);
          t2 = t2.multiplyUnsafe(x).negate();
          t3 = t3.add(t2);
          t3 = t3.subtract(t1);
          const Q = t3.subtract(P);
          return Q;
        },
        fromBytes: pointG2FromBytes,
        toBytes: pointG2ToBytes,
        Signature: {
          fromBytes(bytes) {
            (0, utils_ts_1.abytes)(bytes);
            return signatureG2FromBytes(bytes);
          },
          fromHex(hex) {
            return signatureG2FromBytes(hex);
          },
          toBytes(point) {
            return signatureG2ToBytes(point);
          },
          toRawBytes(point) {
            return signatureG2ToBytes(point);
          },
          toHex(point) {
            return (0, utils_ts_1.bytesToHex)(signatureG2ToBytes(point));
          }
        }
      },
      params: {
        ateLoopSize: BLS_X,
        // The BLS parameter x for BLS12-381
        r: bls12_381_CURVE_G1.n,
        // order; z⁴ − z² + 1; CURVE.n from other curves
        xNegative: true,
        twistType: "multiplicative"
      },
      htfDefaults,
      hash: sha2_js_1.sha256
    });
    var isogenyMapG2 = (0, hash_to_curve_ts_1.isogenyMap)(Fp2, [
      // xNum
      [
        [
          "0x5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97d6",
          "0x5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97d6"
        ],
        [
          "0x0",
          "0x11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71a"
        ],
        [
          "0x11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71e",
          "0x8ab05f8bdd54cde190937e76bc3e447cc27c3d6fbd7063fcd104635a790520c0a395554e5c6aaaa9354ffffffffe38d"
        ],
        [
          "0x171d6541fa38ccfaed6dea691f5fb614cb14b4e7f4e810aa22d6108f142b85757098e38d0f671c7188e2aaaaaaaa5ed1",
          "0x0"
        ]
      ],
      // xDen
      [
        [
          "0x0",
          "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa63"
        ],
        [
          "0xc",
          "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa9f"
        ],
        ["0x1", "0x0"]
        // LAST 1
      ],
      // yNum
      [
        [
          "0x1530477c7ab4113b59a4c18b076d11930f7da5d4a07f649bf54439d87d27e500fc8c25ebf8c92f6812cfc71c71c6d706",
          "0x1530477c7ab4113b59a4c18b076d11930f7da5d4a07f649bf54439d87d27e500fc8c25ebf8c92f6812cfc71c71c6d706"
        ],
        [
          "0x0",
          "0x5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97be"
        ],
        [
          "0x11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71c",
          "0x8ab05f8bdd54cde190937e76bc3e447cc27c3d6fbd7063fcd104635a790520c0a395554e5c6aaaa9354ffffffffe38f"
        ],
        [
          "0x124c9ad43b6cf79bfbf7043de3811ad0761b0f37a1e26286b0e977c69aa274524e79097a56dc4bd9e1b371c71c718b10",
          "0x0"
        ]
      ],
      // yDen
      [
        [
          "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8fb",
          "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8fb"
        ],
        [
          "0x0",
          "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa9d3"
        ],
        [
          "0x12",
          "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa99"
        ],
        ["0x1", "0x0"]
        // LAST 1
      ]
    ].map((i) => i.map((pair) => Fp2.fromBigTuple(pair.map(BigInt)))));
    var isogenyMapG1 = (0, hash_to_curve_ts_1.isogenyMap)(Fp, [
      // xNum
      [
        "0x11a05f2b1e833340b809101dd99815856b303e88a2d7005ff2627b56cdb4e2c85610c2d5f2e62d6eaeac1662734649b7",
        "0x17294ed3e943ab2f0588bab22147a81c7c17e75b2f6a8417f565e33c70d1e86b4838f2a6f318c356e834eef1b3cb83bb",
        "0xd54005db97678ec1d1048c5d10a9a1bce032473295983e56878e501ec68e25c958c3e3d2a09729fe0179f9dac9edcb0",
        "0x1778e7166fcc6db74e0609d307e55412d7f5e4656a8dbf25f1b33289f1b330835336e25ce3107193c5b388641d9b6861",
        "0xe99726a3199f4436642b4b3e4118e5499db995a1257fb3f086eeb65982fac18985a286f301e77c451154ce9ac8895d9",
        "0x1630c3250d7313ff01d1201bf7a74ab5db3cb17dd952799b9ed3ab9097e68f90a0870d2dcae73d19cd13c1c66f652983",
        "0xd6ed6553fe44d296a3726c38ae652bfb11586264f0f8ce19008e218f9c86b2a8da25128c1052ecaddd7f225a139ed84",
        "0x17b81e7701abdbe2e8743884d1117e53356de5ab275b4db1a682c62ef0f2753339b7c8f8c8f475af9ccb5618e3f0c88e",
        "0x80d3cf1f9a78fc47b90b33563be990dc43b756ce79f5574a2c596c928c5d1de4fa295f296b74e956d71986a8497e317",
        "0x169b1f8e1bcfa7c42e0c37515d138f22dd2ecb803a0c5c99676314baf4bb1b7fa3190b2edc0327797f241067be390c9e",
        "0x10321da079ce07e272d8ec09d2565b0dfa7dccdde6787f96d50af36003b14866f69b771f8c285decca67df3f1605fb7b",
        "0x6e08c248e260e70bd1e962381edee3d31d79d7e22c837bc23c0bf1bc24c6b68c24b1b80b64d391fa9c8ba2e8ba2d229"
      ],
      // xDen
      [
        "0x8ca8d548cff19ae18b2e62f4bd3fa6f01d5ef4ba35b48ba9c9588617fc8ac62b558d681be343df8993cf9fa40d21b1c",
        "0x12561a5deb559c4348b4711298e536367041e8ca0cf0800c0126c2588c48bf5713daa8846cb026e9e5c8276ec82b3bff",
        "0xb2962fe57a3225e8137e629bff2991f6f89416f5a718cd1fca64e00b11aceacd6a3d0967c94fedcfcc239ba5cb83e19",
        "0x3425581a58ae2fec83aafef7c40eb545b08243f16b1655154cca8abc28d6fd04976d5243eecf5c4130de8938dc62cd8",
        "0x13a8e162022914a80a6f1d5f43e7a07dffdfc759a12062bb8d6b44e833b306da9bd29ba81f35781d539d395b3532a21e",
        "0xe7355f8e4e667b955390f7f0506c6e9395735e9ce9cad4d0a43bcef24b8982f7400d24bc4228f11c02df9a29f6304a5",
        "0x772caacf16936190f3e0c63e0596721570f5799af53a1894e2e073062aede9cea73b3538f0de06cec2574496ee84a3a",
        "0x14a7ac2a9d64a8b230b3f5b074cf01996e7f63c21bca68a81996e1cdf9822c580fa5b9489d11e2d311f7d99bbdcc5a5e",
        "0xa10ecf6ada54f825e920b3dafc7a3cce07f8d1d7161366b74100da67f39883503826692abba43704776ec3a79a1d641",
        "0x95fc13ab9e92ad4476d6e3eb3a56680f682b4ee96f7d03776df533978f31c1593174e4b4b7865002d6384d168ecdd0a",
        "0x000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"
        // LAST 1
      ],
      // yNum
      [
        "0x90d97c81ba24ee0259d1f094980dcfa11ad138e48a869522b52af6c956543d3cd0c7aee9b3ba3c2be9845719707bb33",
        "0x134996a104ee5811d51036d776fb46831223e96c254f383d0f906343eb67ad34d6c56711962fa8bfe097e75a2e41c696",
        "0xcc786baa966e66f4a384c86a3b49942552e2d658a31ce2c344be4b91400da7d26d521628b00523b8dfe240c72de1f6",
        "0x1f86376e8981c217898751ad8746757d42aa7b90eeb791c09e4a3ec03251cf9de405aba9ec61deca6355c77b0e5f4cb",
        "0x8cc03fdefe0ff135caf4fe2a21529c4195536fbe3ce50b879833fd221351adc2ee7f8dc099040a841b6daecf2e8fedb",
        "0x16603fca40634b6a2211e11db8f0a6a074a7d0d4afadb7bd76505c3d3ad5544e203f6326c95a807299b23ab13633a5f0",
        "0x4ab0b9bcfac1bbcb2c977d027796b3ce75bb8ca2be184cb5231413c4d634f3747a87ac2460f415ec961f8855fe9d6f2",
        "0x987c8d5333ab86fde9926bd2ca6c674170a05bfe3bdd81ffd038da6c26c842642f64550fedfe935a15e4ca31870fb29",
        "0x9fc4018bd96684be88c9e221e4da1bb8f3abd16679dc26c1e8b6e6a1f20cabe69d65201c78607a360370e577bdba587",
        "0xe1bba7a1186bdb5223abde7ada14a23c42a0ca7915af6fe06985e7ed1e4d43b9b3f7055dd4eba6f2bafaaebca731c30",
        "0x19713e47937cd1be0dfd0b8f1d43fb93cd2fcbcb6caf493fd1183e416389e61031bf3a5cce3fbafce813711ad011c132",
        "0x18b46a908f36f6deb918c143fed2edcc523559b8aaf0c2462e6bfe7f911f643249d9cdf41b44d606ce07c8a4d0074d8e",
        "0xb182cac101b9399d155096004f53f447aa7b12a3426b08ec02710e807b4633f06c851c1919211f20d4c04f00b971ef8",
        "0x245a394ad1eca9b72fc00ae7be315dc757b3b080d4c158013e6632d3c40659cc6cf90ad1c232a6442d9d3f5db980133",
        "0x5c129645e44cf1102a159f748c4a3fc5e673d81d7e86568d9ab0f5d396a7ce46ba1049b6579afb7866b1e715475224b",
        "0x15e6be4e990f03ce4ea50b3b42df2eb5cb181d8f84965a3957add4fa95af01b2b665027efec01c7704b456be69c8b604"
      ],
      // yDen
      [
        "0x16112c4c3a9c98b252181140fad0eae9601a6de578980be6eec3232b5be72e7a07f3688ef60c206d01479253b03663c1",
        "0x1962d75c2381201e1a0cbd6c43c348b885c84ff731c4d59ca4a10356f453e01f78a4260763529e3532f6102c2e49a03d",
        "0x58df3306640da276faaae7d6e8eb15778c4855551ae7f310c35a5dd279cd2eca6757cd636f96f891e2538b53dbf67f2",
        "0x16b7d288798e5395f20d23bf89edb4d1d115c5dbddbcd30e123da489e726af41727364f2c28297ada8d26d98445f5416",
        "0xbe0e079545f43e4b00cc912f8228ddcc6d19c9f0f69bbb0542eda0fc9dec916a20b15dc0fd2ededda39142311a5001d",
        "0x8d9e5297186db2d9fb266eaac783182b70152c65550d881c5ecd87b6f0f5a6449f38db9dfa9cce202c6477faaf9b7ac",
        "0x166007c08a99db2fc3ba8734ace9824b5eecfdfa8d0cf8ef5dd365bc400a0051d5fa9c01a58b1fb93d1a1399126a775c",
        "0x16a3ef08be3ea7ea03bcddfabba6ff6ee5a4375efa1f4fd7feb34fd206357132b920f5b00801dee460ee415a15812ed9",
        "0x1866c8ed336c61231a1be54fd1d74cc4f9fb0ce4c6af5920abc5750c4bf39b4852cfe2f7bb9248836b233d9d55535d4a",
        "0x167a55cda70a6e1cea820597d94a84903216f763e13d87bb5308592e7ea7d4fbc7385ea3d529b35e346ef48bb8913f55",
        "0x4d2f259eea405bd48f010a01ad2911d9c6dd039bb61a6290e591b36e636a5c871a5c29f4f83060400f8b49cba8f6aa8",
        "0xaccbb67481d033ff5852c1e48c50c477f94ff8aefce42d28c0f9a88cea7913516f968986f7ebbea9684b529e2561092",
        "0xad6b9514c767fe3c3613144b45f1496543346d98adf02267d5ceef9a00d9b8693000763e3b90ac11e99b138573345cc",
        "0x2660400eb2e4f3b628bdd0d53cd76f2bf565b94e72927c1cb748df27942480e420517bd8714cc80d1fadc1326ed06f7",
        "0xe0fa1d816ddc03e6b24255e0d7819c171c40f65e273b853324efcd6356caa205ca2f570f13497804415473a1d634b8f",
        "0x000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"
        // LAST 1
      ]
    ].map((i) => i.map((j) => BigInt(j))));
    var G1_SWU = (0, weierstrass_ts_1.mapToCurveSimpleSWU)(Fp, {
      A: Fp.create(BigInt("0x144698a3b8e9433d693a02c96d4982b0ea985383ee66a8d8e8981aefd881ac98936f8da0e0f97f5cf428082d584c1d")),
      B: Fp.create(BigInt("0x12e2908d11688030018b12e8753eee3b2016c1f0f24f4070a0b9c14fcef35ef55a23215a316ceaa5d1cc48e98e172be0")),
      Z: Fp.create(BigInt(11))
    });
    var G2_SWU = (0, weierstrass_ts_1.mapToCurveSimpleSWU)(Fp2, {
      A: Fp2.create({ c0: Fp.create(_0n), c1: Fp.create(BigInt(240)) }),
      // A' = 240 * I
      B: Fp2.create({ c0: Fp.create(BigInt(1012)), c1: Fp.create(BigInt(1012)) }),
      // B' = 1012 * (1 + I)
      Z: Fp2.create({ c0: Fp.create(BigInt(-2)), c1: Fp.create(BigInt(-1)) })
      // Z: -(2 + I)
    });
    function mapToG1(scalars) {
      const { x, y } = G1_SWU(Fp.create(scalars[0]));
      return isogenyMapG1(x, y);
    }
    function mapToG2(scalars) {
      const { x, y } = G2_SWU(Fp2.fromBigTuple(scalars));
      return isogenyMapG2(x, y);
    }
  }
});

// scripts/89_verify_drand_receipt.mjs
var import_bls12_381 = __toESM(require_bls12_381(), 1);
var import_sha2 = __toESM(require_sha2(), 1);
var PUBLIC_KEY = "868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31";
function fail(message2) {
  process.stderr.write(`${message2}
`);
  process.exit(2);
}
if (process.argv.length !== 6) fail("usage: verifier ROUND RANDOMNESS SIGNATURE PREVIOUS_SIGNATURE");
var [roundRaw, randomness, signature, previous] = process.argv.slice(2);
if (!/^[1-9][0-9]*$/.test(roundRaw)) fail("invalid round");
if (!/^[0-9a-f]{64}$/.test(randomness)) fail("invalid randomness");
if (!/^[0-9a-f]{192}$/.test(signature) || !/^[0-9a-f]{192}$/.test(previous)) {
  fail("invalid signature encoding");
}
var round = BigInt(roundRaw);
if (round > 0xffffffffffffffffn) fail("round exceeds uint64");
var roundBytes = new Uint8Array(8);
new DataView(roundBytes.buffer).setBigUint64(0, round, false);
var previousBytes = Uint8Array.from(Buffer.from(previous, "hex"));
var messageInput = new Uint8Array(previousBytes.length + roundBytes.length);
messageInput.set(previousBytes, 0);
messageInput.set(roundBytes, previousBytes.length);
var message = (0, import_sha2.sha256)(messageInput);
var derived = Buffer.from((0, import_sha2.sha256)(Uint8Array.from(Buffer.from(signature, "hex")))).toString("hex");
if (derived !== randomness) fail("randomness does not derive from signature");
var verified = false;
try {
  verified = await import_bls12_381.bls12_381.verify(signature, message, PUBLIC_KEY);
} catch (error) {
  fail(`BLS verification error: ${error.message}`);
}
if (!verified) fail("BLS signature verification failed");
process.stdout.write("DRAND_BLS_RECEIPT_OK\n");
