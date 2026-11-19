"""Zero-seed IPC resolver: steamclient64.dll in, ipc/steamclient/<sha>.toml out.

Emits the IPC TOML schema the client parses in IPCLoader.cpp:
per-interface [Name] (interface_id, vtable_rva) plus per-method
[Name.Method] (method_index, funcHash, wrapper_rva, fencepost, argc).

funcHash / fencepost are wire constants the wrapper writes into the outgoing
IPC buffer. They are NOT build-stable and stable/beta re-roll them
independently, so they are read out of the wrapper body on every run - there
is no hand-maintained constant table to go stale. A 4-byte append always has
this shape:

    lea  rdx, [buf]
    mov  r8d, 4                      ; 41 B8 04 00 00 00
    mov  dword ptr [buf], imm32      ; C7 /0 imm32
    lea  rcx, [bufobj]
    call <append helper>

The same shape also carries dynamic values (89 /r stores a register, not an
immediate), and 8-byte appends (41 B8 08 ...) carry neither constant, so the
scan keeps only immediate stores behind a 4-byte append. Exactly two must
appear, in order: first funcHash, second fencepost. Anything else FAILs
loudly - never guess.

Resolution chain per interface, all asserted unique (FAIL loudly):

  RTTI TypeDescriptor name (pinned, e.g. .?AVIClientUserMap@@)
    -> Complete Object Locator (sig == 1, pTD == TD)
      -> vtable: the address X whose qword at X-8 == COL VA
        -> wrapper: qword at X + method_index * 8, must be a pdata entry
           whose body carries the funcHash/fencepost pair

Usage:
    python3 tools/resolve_ipc.py resolve --dll <dll> --out-dir ipc/
        [--channel stable] [--verify <ground-truth.toml>]
"""

import argparse
import struct
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "rp", Path(__file__).parent / "resolve_patterns.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

# mov r8d, 4 - the append length that carries a 4-byte wire constant.
APPEND_4_PREFIX = bytes([0x41, 0xB8, 0x04, 0x00, 0x00, 0x00])

# Used only when the exception directory has no range for a wrapper.
SCAN_LIMIT = 0x400

# A wrapper body can be split across contiguous pdata entries; merging follows
# at most this many links so a corrupt table cannot walk the whole directory.
SCAN_MERGE_LIMIT = 8

# iface -> interface_id, RTTI TypeDescriptor name, and methods:
# (method, method_index, argc). Method indexes are wire-stable; the
# funcHash/fencepost pair is read from the wrapper on every build.
IPC_IFACES = {
    "IClientUser": {
        "id": 1,
        "td": b".?AVIClientUserMap@@",
        "methods": [
            ("GetSteamID", 10, 0),
            ("GetAppOwnershipTicketExtendedData", 105, 2),
            ("RequestEncryptedAppTicket", 120, 2),
            ("GetEncryptedAppTicket", 121, 1),
        ],
    },
    "IClientUtils": {
        "id": 4,
        "td": b".?AVIClientUtilsMap@@",
        "methods": [
            ("GetAppID", 19, 0),
            ("GetAPICallResult", 24, 3),
        ],
    },
}


def image_base(buf) -> int:
    e = struct.unpack_from("<I", buf, 0x3C)[0]
    return struct.unpack_from("<Q", buf, e + 48)[0]


def data_spans(img):
    """(rva, size) of data sections that can host RTTI (order-stable)."""
    return [(va, max(vsz, rawsz)) for name, va, vsz, _raw, rawsz in img.secs
            if name in (b".rdata", b".data")]


def span_bytes(img, rva, size):
    """Contiguous bytes for [rva, rva+size), crossing sections if needed."""
    out = bytearray()
    cur = rva
    while len(out) < size:
        o = img.rva2off(cur)
        if o is None:
            return None
        end = cur + 1
        while end < rva + size and img.rva2off(end) == o + (end - cur):
            end += 1
        out += bytes(img.buf[o:o + (end - cur)])
        cur = end
    return bytes(out)


def _u32(img, rva):
    o = img.rva2off(rva)
    if o is None:
        return None
    return struct.unpack_from("<I", img.buf, o)[0]


def _u64(img, rva):
    o = img.rva2off(rva)
    if o is None:
        return None
    return struct.unpack_from("<Q", img.buf, o)[0]


def pdata_ranges(img):
    """func RVA -> end RVA, straight from the exception directory."""
    off = img.rva2off(img.exc_rva)
    if off is None:
        return {}
    out = {}
    for i in range(img.exc_sz // 12):
        begin, end, _unwind = struct.unpack_from("<III", img.buf, off + i * 12)
        if begin:
            out[begin] = end
    return out


def find_td(img, spans, name: bytes):
    """RVA of the TypeDescriptor whose mangled name matches (name at +16)."""
    cands = set()
    for rva, size in spans:
        blob = span_bytes(img, rva, size)
        if blob is None:
            continue
        off = 0
        key = bytes(name) + b"\x00"
        while True:
            i = blob.find(key, off)
            if i < 0:
                break
            cands.add(rva + i - 16)
            off = i + 1
    if len(cands) != 1:
        return None
    return next(iter(cands))


def find_vtable(img, spans, td_rva, base):
    """The one address X whose qword at X-8 is the VA of a COL for td."""
    cols = set()
    for rva, size in spans:
        blob = span_bytes(img, rva, size)
        if blob is None:
            continue
        for i in range(0, len(blob) - 4, 4):
            if struct.unpack_from("<I", blob, i)[0] != td_rva:
                continue
            col = rva + i - 12
            if _u32(img, col) == 1:
                cols.add(col)
    if not cols:
        return None, "0 COL cands"
    want = {base + c for c in cols}
    match = set()
    for rva, size in spans:
        blob = span_bytes(img, rva, size)
        if blob is None:
            continue
        for i in range(0, len(blob) - 8, 8):
            if struct.unpack_from("<Q", blob, i)[0] in want:
                match.add(rva + i + 8)
    if len(match) != 1:
        return None, f"{len(match)} vtable cands"
    return next(iter(match)), None


def imm32_store(body, i):
    """Decode a C7 /0 memory-immediate store at body[i]; return imm32 or None.

    A register-direct operand (mod == 3) is not a memory store and is
    rejected: the wire constants are always written into the outgoing-buffer
    slot, never into a register.
    """
    if i + 2 > len(body) or body[i] != 0xC7:
        return None
    modrm = body[i + 1]
    mod, rm = modrm >> 6, modrm & 7
    if mod == 3:
        return None
    n = i + 2
    if rm == 4:
        if n >= len(body):
            return None
        sib = body[n]
        n += 1
        if mod == 0 and (sib & 7) == 5:
            n += 4
    elif rm == 5 and mod == 0:
        n += 4
    if mod == 1:
        n += 1
    elif mod == 2:
        n += 4
    if n + 4 > len(body):
        return None
    return int.from_bytes(body[n:n + 4], "little")


def read_append_consts(img, func_rva, func_end, pdata, pends=None):
    """Read (funcHash, fencepost) out of a wrapper body.

    Returns ((funchash, fencepost), None) or (None, reason). Never guesses:
    anything other than exactly two immediate 4-byte appends is a failure.

    A wrapper is sometimes split across several contiguous pdata entries - the
    slot points at the head, and the appended constants live in the next piece
    (IClientUser::GetSteamID did exactly this in beta 1795048523). Contiguous
    entries are one body, so they are read as one.
    """
    if func_rva not in pdata:
        return None, f"wrapper {func_rva:#x} is not a pdata entry"
    o = img.rva2off(func_rva)
    if o is None:
        return None, f"wrapper {func_rva:#x} has no file offset"
    # Only a piece that does not end in RET can be the front half of a split
    # body: a complete wrapper always returns, and its neighbours are unrelated
    # functions whose own appends must never be counted here.
    end = func_end
    for _ in range(SCAN_MERGE_LIMIT):
        tail_off = img.rva2off(max(end, func_rva + 1) - 1)
        if tail_off is None or img.buf[tail_off] == 0xC3:
            break
        nxt = (pends or {}).get(end)
        if nxt is None or nxt <= end:
            break
        end = nxt
    last = max(end, func_rva + 1) - 1
    end_off = img.rva2off(last)
    body = bytes(img.buf[o:(end_off + 1) if end_off is not None
                         else o + SCAN_LIMIT])

    consts = []
    i = body.find(APPEND_4_PREFIX)
    while i >= 0:
        value = imm32_store(body, i + len(APPEND_4_PREFIX))
        if value is not None:
            consts.append(value)
        i = body.find(APPEND_4_PREFIX, i + len(APPEND_4_PREFIX))

    if len(consts) != 2:
        seen = ", ".join(f"{c:#010x}" for c in consts) or "none"
        return None, ("expected 2 immediate 4-byte appends "
                      f"(funcHash, fencepost), found {len(consts)}: {seen}")
    funchash, fencepost = consts
    if funchash == fencepost:
        return None, f"funcHash and fencepost are both {funchash:#010x}"
    return (funchash, fencepost), None


def resolve_iface(spec, img, spans, base, pdata, pends):
    td = find_td(img, spans, spec["td"])
    if td is None:
        return None, None, f'no TD {spec["td"]!r}'
    vt, err = find_vtable(img, spans, td, base)
    if err is not None:
        return None, None, err
    methods = {}
    for name, idx, _argc in spec["methods"]:
        slot = _u64(img, vt + idx * 8)
        if slot is None:
            return None, None, f"{name}: slot {idx} unreadable"
        func_rva = slot - base
        pair, err = read_append_consts(
            img, func_rva, pends.get(func_rva, func_rva + SCAN_LIMIT), pdata,
            pends)
        if err is not None:
            return None, None, f"{name}: {err}"
        methods[name] = (func_rva, pair[0], pair[1], idx)
    return vt, methods, None


def render_toml(resolved) -> str:
    parts = []
    for iface, spec, vt, methods in resolved:
        parts.append("\n".join(
            [f"[{iface}]", f'interface_id = {spec["id"]}',
             f'vtable_rva = "0x{vt:X}"']))
        for name, idx, argc in spec["methods"]:
            wrapper, funchash, fence, actual_idx = methods[name]
            parts.append("\n".join(
                [f"[{iface}.{name}]", f"method_index = {actual_idx}",
                 f'funcHash = "0x{funchash:08X}"',
                 f'wrapper_rva = "0x{wrapper:X}"',
                 f'fencepost = "0x{fence:08X}"', f"argc = {argc}"]))
    head = "# MACHINE-GENERATED by tools/resolve_ipc.py - do not hand-edit"
    return head + "\n" + "\n\n".join(parts) + "\n"


def cmd_resolve(args):
    dll = Path(args.dll)
    img = rp.mp.Image(dll)
    spans = data_spans(img)
    if not spans:
        print("FAIL: no data sections")
        return 1
    base = image_base(img.buf)
    pends = pdata_ranges(img)
    pdata = set(pends)
    if args.channel:
        print(f"channel {args.channel}: wire constants are read per build, "
              "the channel no longer selects a table")
    resolved, failed = [], []
    for iface, iface_spec in IPC_IFACES.items():
        vt, methods, err = resolve_iface(
            iface_spec, img, spans, base, pdata, pends)
        if err is not None:
            failed.append((iface, err))
            print(f"FAIL {iface}: {err}")
        else:
            resolved.append((iface, iface_spec, vt, methods))
            print(f"{iface}: vtable=0x{vt:X} " +
                  " ".join(f"{m}=0x{w[0]:X}" for m, w in methods.items()))
    if args.verify:
        truth = tomllib.load(open(args.verify, "rb"))
        for iface, iface_spec, vt, methods in resolved:
            t = truth.get(iface, {})
            want_vt = t.get("vtable_rva")
            mark = "OK" if want_vt == f"0x{vt:X}" else \
                f"MISMATCH truth={want_vt}"
            print(f"{mark} {iface}.vtable_rva")
            for name, _idx, _argc in iface_spec["methods"]:
                tm = t.get(name, {})
                if not isinstance(tm, dict):
                    print(f"MISSING {iface}.{name} in truth")
                    continue
                wrapper, funchash, fence, _ = methods[name]
                for label, got, want in (
                    ("wrapper_rva", f"0x{wrapper:X}", tm.get("wrapper_rva")),
                    ("funcHash", f"0x{funchash:08X}", tm.get("funcHash")),
                    ("fencepost", f"0x{fence:08X}", tm.get("fencepost")),
                ):
                    mark = "OK" if got == want else f"MISMATCH truth={want}"
                    print(f"{mark} {iface}.{name}.{label}")
    if args.out_dir and not failed:
        from hashlib import sha256
        text = render_toml(resolved)
        stamp = sha256(dll.read_bytes()).hexdigest()
        out = Path(args.out_dir) / "steamclient" / f"{stamp}.toml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, newline="\n")
        print(f"wrote {out}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--dll", required=True)
    r.add_argument("--channel", default=None,
                   help="only labels the log; constants are read per build")
    r.add_argument("--out-dir", default=None)
    r.add_argument("--verify", default=None)
    args = ap.parse_args()
    if args.cmd == "resolve":
        return cmd_resolve(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
