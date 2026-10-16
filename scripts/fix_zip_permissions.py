"""Give a Windows-built deployment zip Linux file permissions.

Windows zips carry no Unix mode bits, so Linux extracts executables
(e.g. playwright/driver/node) without +x. Mark every entry as a Unix
file: 0755 for ELF binaries, shebang scripts and bin/*, 0644 otherwise.

Only needed when deploying from Windows. Run it after the first
`agentcore deploy` has built the dependency cache, then deploy again:

  python scripts/fix_zip_permissions.py starter/.bedrock_agentcore/<agent_name>/dependencies.zip

The original zip is kept next to it as dependencies.zip.bak.
"""
import shutil, sys, zipfile
from pathlib import Path

src = Path(sys.argv[1])
backup = src.with_suffix(".zip.bak")
if not backup.exists():
    shutil.copy2(src, backup)
tmp = src.with_suffix(".zip.tmp")

made_exec = []
with zipfile.ZipFile(backup) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        data = zin.read(info)
        is_dir = info.filename.endswith("/")
        executable = (
            not is_dir
            and (data[:4] == b"\x7fELF" or data[:2] == b"#!" or info.filename.startswith("bin/"))
        )
        mode = 0o040755 if is_dir else (0o100755 if executable else 0o100644)
        new = zipfile.ZipInfo(info.filename, date_time=info.date_time)
        new.compress_type = zipfile.ZIP_DEFLATED
        new.create_system = 3  # Unix, so extractors honour the mode bits
        new.external_attr = mode << 16
        zout.writestr(new, data)
        if executable:
            made_exec.append(info.filename)

tmp.replace(src)
print(f"{len(made_exec)} files marked executable")
for name in made_exec:
    if "playwright/driver" in name or name.startswith("bin/opentelemetry"):
        print("  ", name)
