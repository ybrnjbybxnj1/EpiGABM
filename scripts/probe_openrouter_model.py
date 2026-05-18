
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
from openai import OpenAI

from src.agents.base import parse_decision
from src.agents.prompts import ArchetypeConfig, build_prompt
from src.models.data_structures import EpidemicContext

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _example_prompt() -> str:
    archetype = ArchetypeConfig(
        id="middle_family",
        name="Middle-aged Parent",
        age_range=(30, 55),
        occupation="office worker, teacher, or engineer",
        health_literacy=6,
        risk_tolerance=4,
        family="lives with spouse and children",
        prompt_additions=(
            "Your children go to school. You worry about bringing the flu "
            "home to them. You balance work obligations with family safety."
        ),
    )
    context = EpidemicContext(
        day=10,
        total_infected=120,
        total_susceptible=2700,
        total_recovered=180,
        growth_rate=0.18,
        new_infections_today=24,
        phase="GROWTH",
    )
    return build_prompt(archetype, context, "healthy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str,
                        help="OpenRouter model id, e.g. 'deepseek/deepseek-r1:free'")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in env / .env", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/ybrnjbybxnj1/nir4",
            "X-Title": "EpiGABM-probe",
        },
        timeout=90,
    )

    prompt = _example_prompt()
    if args.show_prompt:
        print("PROMPT:")
        print(prompt)

    print(f"PROBING: {args.model}")
    print(f"  temperature={args.temperature}  max_tokens={args.max_tokens}")

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        print(f"REQUEST FAILED: {type(exc).__name__}: {exc}")
        return 3

    msg = response.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning", None) or ""
    finish_reason = response.choices[0].finish_reason
    usage = getattr(response, "usage", None)

    print(f"\nfinish_reason: {finish_reason}")
    if usage is not None:
        print(f"prompt_tokens: {getattr(usage, 'prompt_tokens', '?')}, "
              f"completion_tokens: {getattr(usage, 'completion_tokens', '?')}, "
              f"total_tokens: {getattr(usage, 'total_tokens', '?')}")

    print(f"\nmessage.content (len={len(content)})")
    print(content if content else "<EMPTY>")

    print(f"\nmessage.reasoning (len={len(reasoning)})")
    if reasoning:
        snippet = reasoning if len(reasoning) <= 1500 else (
            reasoning[:1500] + f"\n... [truncated, total {len(reasoning)} chars]"
        )
        print(snippet)
    else:
        print("<NOT PRESENT or EMPTY>")

    extra_fields = {
        k: v for k, v in dict(msg).items()
        if k not in {"content", "reasoning", "role", "tool_calls", "function_call",
                     "audio", "annotations", "refusal"}
        and v is not None
    }
    if extra_fields:
        print(f"\nadditional message fields")
        print(json.dumps(extra_fields, indent=2, default=str))

    print("\n" + "=" * 78)
    print("PARSE ATTEMPT (with hardened parser)")

    parse_targets = []
    if content:
        parse_targets.append(("content", content))
    if reasoning and not content:
        parse_targets.append(("reasoning", reasoning))
    if reasoning and content:
        parse_targets.append(("reasoning+content", reasoning + "\n" + content))

    if not parse_targets:
        print("nothing to parse: both content and reasoning are empty")
        return 4

    for label, raw in parse_targets:
        print(f"\n  parsing from: {label} (len={len(raw)})")
        try:
            decision = parse_decision(raw)
            print(f"    OK -> isolate={decision.isolate} "
                  f"mask={decision.mask} "
                  f"reduce_contacts={decision.reduce_contacts} "
                  f"see_doctor={decision.see_doctor}")
            print(f"    reasoning field in decision: "
                  f"{decision.reasoning[:120]}{'...' if len(decision.reasoning) > 120 else ''}")
            return 0
        except ValueError as exc:
            print(f"    FAILED: {exc}")

    print("\nALL parse attempts failed.")
    return 5


if __name__ == "__main__":
    sys.exit(main())
