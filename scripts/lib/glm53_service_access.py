"""Startup-only service credential and artifact-access gate; no model imports.

Run inside the actual service, after the supervisor binds this helper and the
manifest hashes. POSIX effective access checks include ACLs. Configuration
declarations, mode bits alone, and checks as the build owner are insufficient.
This component is not yet wired into the serving lifecycle.
"""
import os
from pathlib import Path
import stat

from glm53_contract import sha256_file, strict_json, verify_inventory


def read_credentials():
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in values:
            raise ValueError("duplicate process status field")
        values[key] = value.strip()
    return values


def validate_credentials(values, uid, gid):
    if type(uid) is not int or type(gid) is not int or uid <= 0 or gid <= 0:
        raise ValueError("service identity must be a non-root uid/gid")
    try:
        for field, expected in (("Uid", uid), ("Gid", gid)):
            if [int(item) for item in values[field].split()] != [expected] * 4:
                raise ValueError("service " + field + " identity mismatch")
        if not {int(item) for item in values["Groups"].split()} <= {gid}:
            raise ValueError("service supplementary groups are not empty")
        for field in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
            if int(values[field], 16) != 0:
                raise ValueError("service retains " + field)
        if values["NoNewPrivs"] != "1":
            raise ValueError("service must enforce NoNewPrivileges")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("missing or malformed service credentials") from error


def _identity(path):
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def check_tree_access(root, uid):
    """Check ancestors first, then the complete tree using the effective UID.

    The public gate validates the effective credentials before calling this.
    Reject service ownership even when currently read-only: an owner can chmod.
    """
    root = Path(os.path.abspath(root))
    observed = {}

    def check(path, ancestor=False):
        value = path.lstat()
        if not (stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode)):
            raise ValueError("symlink or non-regular protected path: " + str(path))
        if value.st_uid == uid:
            raise ValueError("service owns protected path: " + str(path))
        if os.access(path, os.W_OK, effective_ids=True):
            raise ValueError("service can write protected path: " + str(path))
        required = os.X_OK if ancestor else os.R_OK | (os.X_OK if stat.S_ISDIR(value.st_mode) else 0)
        if not os.access(path, required, effective_ids=True):
            raise ValueError("service lacks read/search access: " + str(path))
        observed[str(path)] = _identity(path)

    for path in reversed(root.parents):
        check(path, ancestor=True)
    check(root)
    if root.is_dir():
        for path in root.rglob("*"):
            check(path)
    return observed


def reject_writable_descriptors(protected):
    """A preopened writable fd can bypass later file/ACL permission changes."""
    for entry in Path("/proc/self/fd").iterdir():
        try:
            value = os.fstat(int(entry.name))
            if (value.st_dev, value.st_ino) not in protected:
                continue
            info = (Path("/proc/self/fdinfo") / entry.name).read_text()
        except FileNotFoundError:
            continue
        except OSError as error:
            # The directory scan's own descriptor may close before fstat.
            if error.errno == 9:
                continue
            raise
        flags = [line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("flags:")]
        if len(flags) != 1 or int(flags[0], 8) & os.O_ACCMODE != os.O_RDONLY:
            raise ValueError("writable or unidentified protected file descriptor: " + entry.name)


def verify_service_tree(root, manifest_path, manifest_sha256, uid, gid):
    """Validate actual service credentials, immutable access and closed bytes.

    The supervisor supplies the already frozen manifest hash. Revalidation
    brackets hashing; service ownership, ACL write grants, symlinks and inherited
    writable descriptors fail before the caller may allocate model weights.
    """
    validate_credentials(read_credentials(), uid, gid)
    root, manifest_path = Path(root), Path(manifest_path)
    manifest_before = check_tree_access(manifest_path, uid)
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("service inventory manifest hash mismatch")
    before = check_tree_access(root, uid)
    protected = {(row[0], row[1]) for row in (*before.values(), *manifest_before.values())}
    reject_writable_descriptors(protected)
    identities = verify_inventory(root, strict_json(manifest_path))
    if before != check_tree_access(root, uid) or manifest_before != check_tree_access(manifest_path, uid):
        raise ValueError("protected paths changed during service admission")
    reject_writable_descriptors(protected)
    return {"uid": uid, "gid": gid, "files": len(identities), "readonly_access": True,
            "inventory_sha256": manifest_sha256, "qualification": "service_access_only"}
