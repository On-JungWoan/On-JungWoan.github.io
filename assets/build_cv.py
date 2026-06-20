from pathlib import Path
import shutil
import subprocess
import sys

# conda activate cv 먼저 해야 함!!

ROOT = Path(__file__).resolve().parent
TEX = "cv.tex"
ENGINE = "xelatex"
TEX_ENGINE = shutil.which(ENGINE) or Path(sys.executable).with_name("Scripts") / f"{ENGINE}.exe"


def compile_tex() -> None:
    subprocess.run(
        [str(TEX_ENGINE), "-interaction=nonstopmode", "-halt-on-error", TEX],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    compile_tex()
    compile_tex()

    for ext in ("aux", "log", "out", "xdv"):
        (ROOT / f"cv.{ext}").unlink(missing_ok=True)
