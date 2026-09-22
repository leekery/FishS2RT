"""Build a platform-tagged wheel only when a staged worker is present."""

from pathlib import Path

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


HERE = Path(__file__).resolve().parent
WORKER = HERE / "src" / "fishs2rt_runtime" / "_native" / "audiocpp_cli.exe"
if not WORKER.is_file():
    raise RuntimeError(
        "runtime worker is not staged; run scripts/build_runtime_wheel.ps1 "
        "with a verified audiocpp_cli.exe"
    )
for pattern in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "cufft64_*.dll"):
    if len(list(WORKER.parent.glob(pattern))) != 1:
        raise RuntimeError(f"runtime wheel requires exactly one {pattern} beside the worker")
for name in ("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "VCOMP140.DLL"):
    if not (WORKER.parent / name).is_file():
        raise RuntimeError(f"runtime wheel requires {name} beside the worker")


class WindowsRuntimeWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        return "py3", "none", "win_amd64"


setup(cmdclass={"bdist_wheel": WindowsRuntimeWheel})
