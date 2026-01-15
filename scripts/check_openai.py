#!/usr/bin/env python
"""
Minimal OpenAI connectivity + environment sanity check.

This script is intentionally small and dependency-light:
- prints the Python executable and openai package version + path
- checks that OPENAI_API_KEY is set (without printing it)
- performs a lightweight API call:
  - models.list()
  - optional: a tiny chat.completions.create() call

Usage (PowerShell):
  python scripts/check_openai.py
  python scripts/check_openai.py --model gpt-4o-mini
  python scripts/check_openai.py --model gpt-5.2
"""

import argparse
import os
import sys
import site


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenAI environment + connectivity")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable name for API key")
    parser.add_argument("--model", type=str, default=None, help="Optional model name to test a tiny chat completion")
    parser.add_argument("--max-completion-tokens", type=int, default=5, help="max_completion_tokens for chat test")
    parser.add_argument(
        "--no-user-site",
        action="store_true",
        help="Remove user-site packages from sys.path before importing openai (helps if openai==0.x shadows conda/venv).",
    )
    args = parser.parse_args()

    print("python_executable:", sys.executable)

    if args.no_user_site:
        try:
            user_site = site.getusersitepackages()
        except Exception:
            user_site = None
        if user_site:
            sys.path = [p for p in sys.path if not (isinstance(p, str) and (p == user_site or p.startswith(user_site + os.sep)))]
            print("user_site_removed:", user_site)
        else:
            print("user_site_removed:", None)

    try:
        import openai  # noqa: F401
        import importlib.metadata as md

        try:
            v = md.version("openai")
        except Exception:
            v = getattr(openai, "__version__", "unknown")
        print("openai_version:", v)
        print("openai_module_path:", getattr(openai, "__file__", "unknown"))
        has_new = hasattr(openai, "OpenAI")
        print("openai_has_OpenAI_client:", has_new)
    except Exception as e:
        print("ERROR: cannot import openai:", repr(e))
        return 2

    api_key = os.getenv(args.api_key_env)
    print(f"{args.api_key_env}_set:", bool(api_key))
    if not api_key:
        print(f"ERROR: {args.api_key_env} is not set in this shell.")
        return 3

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        # 1) models.list()
        models = client.models.list()
        first_id = models.data[0].id if models and models.data else None
        print("models_list_ok:", True)
        print("models_list_first_id:", first_id)

        if args.model:
            # 2) tiny chat completion
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=args.max_completion_tokens,
            )
            out = r.choices[0].message.content if r.choices else None
            print("chat_test_ok:", True)
            print("chat_test_output:", out)
    except Exception as e:
        print("ERROR: OpenAI API call failed:", repr(e))
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


