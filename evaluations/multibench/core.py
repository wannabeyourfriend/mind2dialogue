from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_TRAILING_THINK_CLOSE = re.compile(r"</think>\s*")


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    text = _THINK_BLOCK.sub("", text)
    text = _TRAILING_THINK_CLOSE.sub("", text)
    return text.strip()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, p)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    return repo_root() / "data"


def benchmark_data_dir(name: str) -> Path:
    p = data_root() / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing data directory for benchmark '{name}' at {p}. "
            f"Create a symlink pointing to the upstream dataset."
        )
    return p


@dataclass
class LLMClient:
    model_name: str = "default"
    api_base: str = "http://localhost:8000/v1"
    api_key: str | None = None
    max_retries: int = 6
    default_max_tokens: int = 512
    default_temperature: float = 0.0
    strip_think: bool = True

    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = OpenAI(base_url=self.api_base, api_key=key)

    def chat(
        self,
        messages: Sequence[dict] | str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        extra_body: dict | None = None,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        kwargs: dict[str, Any] = {
            "model": model or self.model_name,
            "messages": msgs,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        backoff = 1.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                return strip_think_tags(text) if self.strip_think else text.strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("chat attempt %d/%d failed: %s", attempt + 1, self.max_retries, e)
                if attempt == self.max_retries - 1:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        raise RuntimeError(f"chat failed after {self.max_retries} retries: {last_err}")

    def chat_batch(
        self,
        items: Sequence[Any],
        build_messages: Callable[[Any], Sequence[dict] | str],
        *,
        workers: int = 20,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        desc: str = "chat",
        on_error: str = "store",
    ) -> list[Any]:
        results: list[Any] = [None] * len(items)

        def _one(idx: int) -> tuple[int, Any]:
            try:
                msgs = build_messages(items[idx])
                out = self.chat(
                    msgs,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return idx, out
            except Exception as e:  # noqa: BLE001
                if on_error == "raise":
                    raise
                return idx, e

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(_one, i) for i in range(len(items))]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx, val = fut.result()
                results[idx] = val
        return results


def client_from_args(args) -> LLMClient:
    model_name = getattr(args, "model", None) or getattr(args, "model_name", None) or "default"
    return LLMClient(
        model_name=model_name,
        api_base=getattr(args, "api_base", "http://localhost:8000/v1"),
        api_key=getattr(args, "api_key", None),
        default_max_tokens=getattr(args, "max_tokens", 512),
        default_temperature=getattr(args, "temperature", 0.0),
        strip_think=getattr(args, "strip_think", True),
    )


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("LLM endpoint")
    g.add_argument("--api-base", "--api_base", dest="api_base",
                   default="http://localhost:8000/v1",
                   help="OpenAI-compatible base URL.")
    g.add_argument("--model", "--model-name", "--model_name", dest="model",
                   required=True,
                   help="Model name as served by the vLLM/OpenAI endpoint.")
    g.add_argument("--api-key", "--api_key", dest="api_key", default=None,
                   help="Optional API key (default: $OPENAI_API_KEY or 'not-needed').")
    g.add_argument("--max-tokens", "--max_tokens", dest="max_tokens",
                   type=int, default=512, help="Default max completion tokens.")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="Default sampling temperature.")
    g.add_argument("--no-strip-think", dest="strip_think", action="store_false",
                   default=True, help="Keep <think>...</think> blocks in output.")

    g2 = parser.add_argument_group("Runtime")
    g2.add_argument("--workers", "--parallel", dest="workers", type=int, default=32,
                    help="Parallel request workers (ThreadPoolExecutor).")
    g2.add_argument("--max-items", "--max_items", dest="max_items",
                    type=int, default=None,
                    help="Optional cap on items for quick testing.")
    g2.add_argument("--output-dir", "--output_dir", dest="output_dir",
                    default=None, required=True,
                    help="Directory to write results to.")
    g2.add_argument("--seed", type=int, default=0, help="Random seed where used.")


BENCHMARKS = [
    "bigtom",
    "personamem",
    "prefeval",
]


def _load(bench: str) -> ModuleType:
    try:
        return importlib.import_module(f"multibench.benchmarks.{bench}.run")
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Unknown or unimportable benchmark '{bench}'. Known: {BENCHMARKS}\n"
            f"Import error: {e}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="multibench",
        description="Run personalization benchmarks against an OpenAI-compatible endpoint.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single benchmark.")
    p_run.add_argument("benchmark", choices=BENCHMARKS)
    p_run.add_argument("rest", nargs=argparse.REMAINDER,
                       help="Arguments forwarded to the benchmark runner (use -- to separate).")

    sub.add_parser("list", help="List available benchmarks.")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        print("\n".join(BENCHMARKS))
        return 0

    if args.cmd == "run":
        mod = _load(args.benchmark)
        forwarded = args.rest
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        if not hasattr(mod, "main"):
            raise SystemExit(f"Benchmark '{args.benchmark}' has no main() entry point.")
        rc = mod.main(forwarded)
        return rc or 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
