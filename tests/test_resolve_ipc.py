import importlib.util
import struct
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ri", Path(__file__).parent / ".." / "tools" / "resolve_ipc.py")
ri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ri)


class FlatImg:
    """Flat addressable image: put() then rva2off()/buf work on real RVAs."""

    def __init__(self, base=0x1000):
        self.base = base
        self.buf = bytearray(0x20000)

    def put(self, rva, data):
        self.buf[rva - self.base:rva - self.base + len(data)] = bytes(data)

    def rva2off(self, rva):
        return rva - self.base


RDATA, RSZ = 0x2000, 0x4000
SPANS = [(RDATA, RSZ)]
TD = b".?AVIClientUserMap@@"
TD_RVA = RDATA + 0x100
COL_RVA = RDATA + 0x200
VT_RVA = 0x5000
WRAP = VT_RVA + 0x1000
BASE = 0x140000000

FUNC_HASH = 0xD6FC3200
FENCEPOST = 0xD7058CA5

# mov r8d, 4 followed by a register store: the append carries a dynamic value,
# never a wire constant, so the reader must skip it.
DYN_APPEND = bytes([0x41, 0xB8, 0x04, 0x00, 0x00, 0x00,
                    0x89, 0x45, 0x40])
# mov r8d, 4 followed by a memory-immediate store.
IMM_APPEND = bytes([0x41, 0xB8, 0x04, 0x00, 0x00, 0x00, 0xC7, 0x45, 0x40])


def imm_append(value):
    return IMM_APPEND + struct.pack("<I", value)


def wrapper_body(funchash=FUNC_HASH, fencepost=FENCEPOST):
    """Prologue, one dynamic append, then the funcHash/fencepost pair."""
    return (b"\x48\x89\x5C\x24\x18\x55\x56\x57\x48\x8B\xEC"
            + DYN_APPEND
            + imm_append(funchash) + imm_append(fencepost)
            + b"\xC3")


def pends_for(body):
    return {WRAP: WRAP + len(body)}


def mkimg(entries=None, with_td=True, with_col=True, col_sig=1,
          vt_slot_val=None, body=None):
    img = FlatImg()
    if with_td:
        img.put(TD_RVA, b"\x00" * 16 + TD + b"\x00")
    if with_col:
        img.put(COL_RVA, struct.pack("<IIIII", col_sig, 0, 0, TD_RVA, 0x9999))
        img.put(VT_RVA - 8, struct.pack("<Q", BASE + COL_RVA))
    if body is None:
        body = wrapper_body()
    img.put(WRAP, body)
    slotval = WRAP if vt_slot_val is None else vt_slot_val
    img.put(VT_RVA + 10 * 8, struct.pack("<Q", BASE + slotval))
    ctx_entries = list(entries) if entries is not None else [WRAP]
    return img, ctx_entries


class TestLocateTd(unittest.TestCase):
    def test_found(self):
        img, _ = mkimg()
        self.assertEqual(ri.find_td(img, SPANS, TD), TD_RVA)

    def test_absent(self):
        img, _ = mkimg(with_td=False)
        self.assertIsNone(ri.find_td(img, SPANS, TD))


class TestLocateVtable(unittest.TestCase):
    def test_found(self):
        img, _ = mkimg()
        self.assertEqual(
            ri.find_vtable(img, SPANS, TD_RVA, BASE),
            (VT_RVA, None))

    def test_no_col_fails(self):
        img, _ = mkimg(with_col=False)
        v, e = ri.find_vtable(img, SPANS, TD_RVA, BASE)
        self.assertIsNone(v)
        self.assertIn("0 COL", e)

    def test_bad_sig_col_ignored(self):
        img, _ = mkimg(col_sig=0)
        v, e = ri.find_vtable(img, SPANS, TD_RVA, BASE)
        self.assertIsNone(v)
        self.assertIn("0 COL", e)

    def test_no_ref_fails(self):
        img, _ = mkimg()
        img.put(VT_RVA - 8, struct.pack("<Q", 0xDEAD))
        v, e = ri.find_vtable(img, SPANS, TD_RVA, BASE)
        self.assertIsNone(v)
        self.assertIn("0 vtable", e)

    def test_two_refs_fail(self):
        img, _ = mkimg()
        img.put(VT_RVA + 0x800 - 8, struct.pack("<Q", BASE + COL_RVA))
        v, e = ri.find_vtable(img, SPANS, TD_RVA, BASE)
        self.assertIsNone(v)
        self.assertIn("2 vtable", e)


class TestReadAppendConsts(unittest.TestCase):
    def test_reads_pair_in_order(self):
        body = wrapper_body()
        img, _ = mkimg(body=body)
        pair, err = ri.read_append_consts(
            img, WRAP, WRAP + len(body), {WRAP})
        self.assertIsNone(err)
        self.assertEqual(pair, (FUNC_HASH, FENCEPOST))

    def test_dynamic_append_is_not_a_constant(self):
        body = wrapper_body()
        img, _ = mkimg(body=body)
        pair, err = ri.read_append_consts(
            img, WRAP, WRAP + len(body), {WRAP})
        self.assertEqual(pair[0], FUNC_HASH)
        self.assertNotEqual(pair[1], 0x00000001)

    def test_not_in_pdata(self):
        body = wrapper_body()
        img, _ = mkimg(body=body)
        pair, err = ri.read_append_consts(
            img, WRAP, WRAP + len(body), {0x1234})
        self.assertIsNone(pair)
        self.assertIn("not a pdata entry", err)

    def test_single_constant_fails(self):
        body = b"\x55" + imm_append(FUNC_HASH) + b"\xC3"
        img, _ = mkimg(body=body)
        pair, err = ri.read_append_consts(
            img, WRAP, WRAP + len(body), {WRAP})
        self.assertIsNone(pair)
        self.assertIn("found 1", err)

    def test_three_constants_fail(self):
        body = b"\x55" + imm_append(1) + imm_append(FUNC_HASH) \
            + imm_append(FENCEPOST) + b"\xC3"
        img, _ = mkimg(body=body)
        pair, err = ri.read_append_consts(
            img, WRAP, WRAP + len(body), {WRAP})
        self.assertIsNone(pair)
        self.assertIn("found 3", err)

    def test_identical_pair_fails(self):
        body = b"\x55" + imm_append(FUNC_HASH) + imm_append(FUNC_HASH) + b"\xC3"
        img, _ = mkimg(body=body)
        pair, err = ri.read_append_consts(
            img, WRAP, WRAP + len(body), {WRAP})
        self.assertIsNone(pair)
        self.assertIn("both", err)

    def test_register_direct_store_rejected(self):
        # C7 C0 imm32 is mov eax, imm32 (mod == 3): not a memory store.
        body = b"\x55" + bytes([0x41, 0xB8, 0x04, 0, 0, 0, 0xC7, 0xC0]) \
            + struct.pack("<I", FUNC_HASH) + b"\xC3"
        img, _ = mkimg(body=body)
        pair, err = ri.read_append_consts(
            img, WRAP, WRAP + len(body), {WRAP})
        self.assertIsNone(pair)
        self.assertIn("found 0", err)


class TestResolveIface(unittest.TestCase):
    SPEC = {"id": 1, "td": TD,
            "methods": [("GetSteamID", 10, 0)]}

    def test_happy_path(self):
        img, entries = mkimg()
        body = wrapper_body()
        vt, methods, err = ri.resolve_iface(
            self.SPEC, img, SPANS, BASE, set(entries), pends_for(body))
        self.assertIsNone(err)
        self.assertEqual(vt, VT_RVA)
        self.assertEqual(methods["GetSteamID"],
                         (WRAP, FUNC_HASH, FENCEPOST, 10))

    def test_bad_slot_fails(self):
        img, entries = mkimg()
        body = wrapper_body()
        img.put(VT_RVA + 10 * 8, struct.pack("<Q", BASE + 0x9999))
        vt, methods, err = ri.resolve_iface(
            self.SPEC, img, SPANS, BASE, set(entries), pends_for(body))
        self.assertIsNone(vt)
        self.assertIn("not a pdata entry", err)

    def test_slot_is_never_probed_sideways(self):
        # The index is authoritative: a wrapper parked one slot over must not
        # be picked up, because a wrong method_index corrupts the client.
        img, entries = mkimg()
        body = wrapper_body()
        img.put(VT_RVA + 10 * 8, struct.pack("<Q", BASE + 0x9999))
        img.put(VT_RVA + 11 * 8, struct.pack("<Q", BASE + WRAP))
        vt, methods, err = ri.resolve_iface(
            self.SPEC, img, SPANS, BASE, set(entries), pends_for(body))
        self.assertIsNone(vt)
        self.assertIsNotNone(err)


class TestRenderToml(unittest.TestCase):
    def test_schema_matches_upstream(self):
        spec = {"id": 1, "td": TD,
                "methods": [("GetSteamID", 10, 0)]}
        text = ri.render_toml([("IClientUser", spec, 0x1324AF0,
                                {"GetSteamID": (0x79F3C0, FUNC_HASH,
                                                FENCEPOST, 10)})])
        self.assertEqual(text, """\
# MACHINE-GENERATED by tools/resolve_ipc.py - do not hand-edit
[IClientUser]
interface_id = 1
vtable_rva = "0x1324AF0"

[IClientUser.GetSteamID]
method_index = 10
funcHash = "0xD6FC3200"
wrapper_rva = "0x79F3C0"
fencepost = "0xD7058CA5"
argc = 0
""")


class TestNoConstantTable(unittest.TestCase):
    def test_channel_table_is_gone(self):
        # The hand-maintained table was the failure mode: it went stale
        # silently. It must not come back.
        self.assertFalse(hasattr(ri, "IPC_CONSTANTS"))


if __name__ == "__main__":
    unittest.main()
