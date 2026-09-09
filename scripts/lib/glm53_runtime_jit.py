"""Explicitly selected sealed Triton cache; never silently compile a miss.

This helper is not yet installed or enabled in a serving runtime. The complete
serving gate also needs guards for native-load fallback, DeepGEMM, CuTe and
other compilation paths. Cache roots must be immutable to the model process.
"""
from pathlib import Path
import hashlib
import re
import stat
import sysconfig

from glm53_contract import strict_json, verify_inventory


def activate_flashinfer(root, manifest, modules, *, enabled=False):
    """Preload frozen Nvcc modules and select a loader with no build fallback.

    Evidence-only startup selection, not installed in the serving runtime.
    The caller freezes runtime/manifest bytes and enforces engine-read-only
    cache access. This does not cover independent CuTe or DeepGEMM entry points.
    Disabled mode performs no imports, filesystem accesses or runtime changes.
    """
    if type(enabled) is not bool:
        raise ValueError('sealed FlashInfer selection must be boolean')
    if not enabled:
        return
    root = Path(root).absolute()
    identities = verify_inventory(root, manifest)
    if not isinstance(modules, dict) or not modules:
        raise ValueError('sealed FlashInfer requires explicit module bindings')
    paths = {}
    for name, relative in modules.items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name) or
                not isinstance(relative, str) or relative != name + '/' + name + '.so' or
                relative not in identities):
            raise ValueError('invalid sealed FlashInfer module binding')
        paths[name] = str(root / relative)
    import flashinfer.jit.core as core
    # Complete every load and revalidation before mutating runtime entry points.
    loaded = {name: core.tvm_ffi.load_module(path) for name, path in paths.items()}
    if any(value is None for value in loaded.values()):
        raise ValueError('sealed FlashInfer loader returned no module')
    if verify_inventory(root, manifest) != identities:
        raise ValueError('sealed FlashInfer cache changed during loading')
    spec_class = core.JitSpecNvcc

    def select(spec, so_path=None):
        if type(spec) is not spec_class or spec.name not in loaded:
            raise ValueError('unlisted sealed FlashInfer module')
        if so_path is not None and str(so_path) != paths[spec.name]:
            raise ValueError('unlisted sealed FlashInfer library path')
        return loaded[spec.name]

    def reject_build(*args, **kwargs):
        raise ValueError('sealed FlashInfer compilation is unavailable')

    def load_path(path):
        for name, frozen_path in paths.items():
            if str(path) == frozen_path:
                return loaded[name]
        raise ValueError('unlisted sealed FlashInfer library path')

    # Original Nvcc.load aliases resolve this core-local namespace. Do not
    # replace the shared tvm_ffi package's loader used by other libraries.
    from types import SimpleNamespace
    core.tvm_ffi = SimpleNamespace(load_module=load_path)
    core.JitSpec.build_and_load = select
    spec_class.try_load = select
    spec_class.load = select
    spec_class.build = reject_build
    spec_class.write_ninja = reject_build
    # Retained aliases of Nvcc.build resolve these globals at call time.
    core.run_ninja = reject_build
    return {'selection': 'sealed_FlashInfer_Nvcc', 'modules': [
        {'name': name, 'path': paths[name]} for name in sorted(paths)]}


def activate_triton(cache_class, *, enabled=False):
    """Startup-only selection; disabled mode does not import or change Triton.

    The lifecycle must log this selection and bind the verified cache before
    any model allocation. This covers Triton only, not all CUDA compilation.
    """
    if type(enabled) is not bool:
        raise ValueError("sealed Triton selection must be boolean")
    if not enabled:
        return
    import triton.knobs as knobs
    import triton.runtime.build as build
    if cache_class is None:
        raise ValueError("sealed Triton mode requires a verified cache class")
    for field in ("always_compile", "override", "dump_ir"):
        if getattr(knobs.compilation, field):
            raise ValueError("sealed Triton rejects " + field)
    if knobs.cache.manager_class is not None or knobs.cache.remote_manager_class is not None:
        raise ValueError("sealed Triton rejects another cache manager")
    if knobs.runtime.add_stages_inspection_hook is not None or knobs.compilation.listener is not None:
        raise ValueError("sealed Triton rejects compiler hooks")

    def reject_compilation(*args, **kwargs):
        raise RuntimeError("sealed Triton native compilation is unavailable")

    # Resolve switches now so later environment lookups cannot enable codegen.
    knobs.compilation.always_compile = False
    knobs.compilation.override = False
    knobs.compilation.dump_ir = False
    knobs.cache.manager_class = cache_class
    knobs.cache.remote_manager_class = None
    build.compile_module_from_src = sealed_native_loader(build)
    # Also close previously imported aliases of the original loader: its
    # fallback resolves _build from this module when it encounters an error.
    build._build = reject_compilation


def sealed_native_loader(build_module):
    """Select a loader without Triton's native-compiler fallback at startup.

    The caller must first install the sealed cache class. Keep the original
    source/platform cache-key formula so prebuilt helper modules are reusable.
    Errors loading a frozen native artifact propagate without recompilation.
    """
    def load(src, name, library_dirs=None, include_dirs=None, libraries=None, ccflags=None):
        key = hashlib.sha256((src + build_module.platform_key()).encode("utf-8")).hexdigest()
        cache = build_module.get_cache_manager(key)
        path = cache.get_file(name + sysconfig.get_config_var("EXT_SUFFIX"))
        if path is None:
            raise ValueError("unsealed native cache miss")
        return build_module._load_module_from_path(name, path)
    return load


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def sealed_cache_class(root, manifest):
    """Verify once before weights load; return a class for TRITON_CACHE_MANAGER.

    The returned class has no environment reads and no cache mutation path.
    Calls occur when loading a specialization, not on a resident kernel hit.
    """
    root = Path(root).absolute()
    identities = verify_inventory(root, manifest)
    directories = {str(p.relative_to(root)): _identity(p.lstat())
                   for p in (root, *root.rglob("*")) if p.is_dir()}
    keys = {name.split("/")[0] for name in identities if "/" in name}

    class SealedCache:
        def __init__(self, key, override=False, dump=False):
            if override or dump:
                raise ValueError("sealed cache rejects override and dump modes")
            if not isinstance(key, str) or not re.fullmatch(r"[A-Z2-7]{1,128}", key):
                raise ValueError("invalid sealed cache key")
            if key not in keys:
                raise ValueError("unknown unsealed kernel specialization")
            self.key = key

        def get_file(self, filename):
            if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename) or filename in (".", ".."):
                raise ValueError("invalid sealed cache path")
            name = self.key + "/" + filename
            if name not in identities:
                raise ValueError("unsealed or missing cache artifact: " + name)
            for directory in (root, root / self.key):
                observed = directory.lstat()
                relative = str(directory.relative_to(root))
                if not stat.S_ISDIR(observed.st_mode) or _identity(observed) != directories.get(relative):
                    raise ValueError("sealed cache directory changed")
            path = root / name
            try:
                observed = path.lstat()
            except FileNotFoundError as exc:
                raise ValueError("missing sealed cache artifact: " + name) from exc
            if not stat.S_ISREG(observed.st_mode) or _identity(observed) != identities[name]:
                raise ValueError("sealed cache artifact changed: " + name)
            return str(path)

        def get_group(self, filename):
            group_name = "__grp__" + filename
            group_path = self.get_file(group_name)
            group = strict_json(Path(group_path))
            if set(group) != {"child_paths"} or not isinstance(group["child_paths"], dict):
                raise ValueError("invalid sealed cache group metadata")
            if filename not in group["child_paths"]:
                raise ValueError("sealed group lacks requested metadata")
            children = {}
            for name, path in group["child_paths"].items():
                actual = self.get_file(name)
                if path != actual:
                    raise ValueError("unsealed group child path")
                children[name] = actual
            self.get_file(group_name)
            return children

        def put(self, data, filename, binary=True):
            raise ValueError("sealed cache cannot compile or write")

        def put_group(self, filename, group):
            raise ValueError("sealed cache cannot compile or write groups")

    return SealedCache
