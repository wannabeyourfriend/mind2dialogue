"""Run the official PersonaMem-v2 evaluator from a user-supplied checkout."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable,
                        help="Python environment containing upstream dependencies")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-mode", choices=["mcq", "generative", "both"], default="mcq")
    parser.add_argument("--size", choices=["32k", "128k"], default="32k")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    upstream = args.upstream_dir.resolve()
    script = upstream / "inference.py"
    if not script.is_file():
        parser.error(f"Official PersonaMem-v2 inference.py not found: {script}")
    command = [
        args.python, str(script),
        "--model_name", args.model,
        "--benchmark_file", str(args.benchmark_file.resolve()),
        "--result_path", str(args.output_dir.resolve()),
        "--eval_mode", args.eval_mode,
        "--size", args.size,
        "--parallel", str(args.workers),
    ]
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = args.api_base
    env.setdefault("OPENAI_API_KEY", "not-needed")
    return subprocess.run(command, cwd=upstream, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
