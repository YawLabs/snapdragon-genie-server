"""Which Hexagon architectures this box will actually offer.

hexagon_search_path is pure filesystem logic, but it has already been wrong in
both directions: first hardcoded to v73, which silently excluded X2 Elite (v81);
then globbing every skel the SDK ships, which offered v75 and v79 -- Android
parts that carry a skel but no Windows stub, so the box cannot reach them. Both
failures are quiet at startup and expensive later: a bundle that will not load,
or a promise the hardware cannot keep.

The rule it encodes is an INTERSECTION -- an arch counts only if the SDK ships
both lib/hexagon-vNN/unsigned and lib/aarch64-windows-msvc/QnnHtpVNNStub.dll --
so every test here is about one half being present without the other.

Device-free: the SDK is a handful of empty files under tmp_path. Nothing is
loaded, only looked up by name.
"""

import os

import pytest


def build_sdk(gs, root, skels=(), stubs=()):
    """Point gs at a fake QAIRT tree: `skels` get a skel dir, `stubs` a DLL.

    Sets BOTH module globals deliberately. LIB_DIR is derived from SDK_DIR at
    import time, not per call, so moving SDK_DIR alone leaves the stub half
    still resolving against whatever GENIE_SDK_DIR the shell happened to export
    -- the test would then be reading the developer's real SDK.
    """
    lib = root / "lib"
    for arch in skels:
        (lib / ("hexagon-%s" % arch) / "unsigned").mkdir(parents=True)
    msvc = lib / "aarch64-windows-msvc"
    msvc.mkdir(parents=True, exist_ok=True)
    for arch in stubs:
        (msvc / ("QnnHtpV%sStub.dll" % arch.lstrip("v"))).write_bytes(b"")
    gs.SDK_DIR = str(root)
    gs.LIB_DIR = str(msvc)


@pytest.fixture
def sdk(gs, tmp_path, monkeypatch):
    """build_sdk bound to this test's gs/tmp_path, with the pin cleared.

    GENIE_HEXAGON_ARCH is read from os.environ on every call, so a developer
    who pinned an arch in their shell would otherwise silence half this file.
    """
    monkeypatch.delenv("GENIE_HEXAGON_ARCH", raising=False)

    def make(skels=(), stubs=()):
        build_sdk(gs, tmp_path, skels=skels, stubs=stubs)
        return tmp_path
    return make


# --- the intersection ------------------------------------------------------

def test_an_arch_with_both_halves_is_usable(sdk, gs):
    sdk(skels=["v73", "v81"], stubs=["v73", "v81"])
    path, usable, skel_only = gs.hexagon_search_path()
    # The v73-hardcoded version lost v81 on an X2 Elite box and said nothing.
    assert usable == ["v73", "v81"]
    assert skel_only == []
    assert path


def test_a_skel_without_a_windows_stub_is_reported_not_offered(sdk, gs):
    # v75 (8 Gen 3) and v79 (8 Elite) are Android parts: QAIRT ships their
    # skels, Windows can never reach them. Offering them put an unloadable arch
    # on ADSP_LIBRARY_PATH; dropping them silently left load_engine's "no usable
    # Hexagon" exit with nothing to name.
    sdk(skels=["v73", "v75", "v79", "v81"], stubs=["v73", "v81"])
    path, usable, skel_only = gs.hexagon_search_path()
    assert usable == ["v73", "v81"]
    assert skel_only == ["v75", "v79"]
    assert "hexagon-v75" not in path and "hexagon-v79" not in path


def test_a_stub_without_a_skel_appears_nowhere(sdk, gs):
    # The other half missing is not a diagnosis worth printing -- there is no
    # skel to point ADSP_LIBRARY_PATH at, and it is not "skel-only" either.
    sdk(skels=["v73"], stubs=["v68", "v73"])
    path, usable, skel_only = gs.hexagon_search_path()
    assert usable == ["v73"]
    assert skel_only == []
    assert "v68" not in usable and "v68" not in skel_only


def test_a_hexagon_dir_with_no_unsigned_subdir_is_not_offered(sdk, gs, tmp_path):
    # The skel half is specifically lib/hexagon-vNN/unsigned. A bare
    # hexagon-vNN dir (signed-only, or a half-extracted SDK) has nothing
    # loadable in it, so counting it would offer an arch with no skels.
    sdk(skels=["v73"], stubs=["v73", "v81"])
    (tmp_path / "lib" / "hexagon-v81").mkdir()
    path, usable, skel_only = gs.hexagon_search_path()
    assert usable == ["v73"]
    assert "v81" not in usable and "v81" not in skel_only


# --- GENIE_HEXAGON_ARCH ----------------------------------------------------

def test_the_pin_selects_exactly_one_of_several_usable_archs(sdk, gs, monkeypatch):
    sdk(skels=["v73", "v81"], stubs=["v73", "v81"])
    monkeypatch.setenv("GENIE_HEXAGON_ARCH", "v81")
    path, usable, skel_only = gs.hexagon_search_path()
    # The escape hatch for forcing an arch is worthless if the other one still
    # rides along on ADSP_LIBRARY_PATH.
    assert usable == ["v81"]
    assert len(path.split(os.pathsep)) == 1 and "hexagon-v81" in path
    assert "hexagon-v73" not in path


@pytest.mark.parametrize("pin", ["v75", "v99"])
def test_a_pin_naming_an_unusable_arch_offers_nothing(sdk, gs, monkeypatch, pin):
    # v75 is skel-only, v99 does not exist. Either way the honest answer is an
    # empty list, which load_engine turns into an exit telling you to unset the
    # pin. Falling back to "some other arch" would run a bundle compiled for the
    # pinned one on hardware it was not built for.
    sdk(skels=["v73", "v75", "v81"], stubs=["v73", "v81"])
    monkeypatch.setenv("GENIE_HEXAGON_ARCH", pin)
    path, usable, skel_only = gs.hexagon_search_path()
    assert usable == []
    assert path == ""
    assert skel_only == ["v75"]     # still says WHY, so the exit can name it


# --- the returned path string ----------------------------------------------

def test_the_path_carries_one_skel_dir_per_usable_arch(sdk, gs):
    # This string becomes ADSP_LIBRARY_PATH verbatim. An entry per usable arch,
    # each pointing at the unsigned dir -- a stray or missing entry is a load
    # failure at dialog-create time, far from here.
    sdk(skels=["v73", "v75", "v81"], stubs=["v73", "v81"])
    path, usable, skel_only = gs.hexagon_search_path()
    entries = path.split(os.pathsep)
    assert len(entries) == len(usable) == 2
    for arch, entry in zip(usable, entries, strict=True):
        assert os.path.basename(entry) == "unsigned"
        assert os.path.basename(os.path.dirname(entry)) == "hexagon-" + arch
        assert os.path.isdir(entry)


def test_an_empty_sdk_reports_nothing_rather_than_raising(gs, tmp_path, monkeypatch):
    # A wrong GENIE_SDK_DIR must reach load_engine's "no usable Hexagon"
    # message, not a glob traceback out of module scope.
    monkeypatch.delenv("GENIE_HEXAGON_ARCH", raising=False)
    gs.SDK_DIR = str(tmp_path)
    gs.LIB_DIR = os.path.join(str(tmp_path), "lib", "aarch64-windows-msvc")
    assert gs.hexagon_search_path() == ("", [], [])
