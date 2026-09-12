#!/usr/bin/env python3
"""Real Qwen serving acceptance, using only the Python standard library.

Run after /health succeeds, with no unrelated traffic for cache metric attribution.
Short mode exercises defaults, tools, concurrency and growing prefix reuse. Long
mode tokenizes and retrieves from >=260000 input tokens at a 262144 context limit.
This is functional evidence, not a speed comparison or a quality benchmark.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
from threading import Barrier
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


CONTEXT = 262144
LONG_OUTPUT = 1024


def message(text):
    return {"role": "user", "content": text}


def json_answer(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18300")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--mode", choices=("short", "long", "all"), default="all")
    args = parser.parse_args()
    args.base = args.base.rstrip("/")
    model = "qwen3.8-flash-next"
    secret = ""
    started = time.monotonic()
    report = {
        "base": args.base, "mode": args.mode, "passed": False, "checks": [],
        "limitations": "Functional checks only; no comparative benchmark. Prefix metrics are server-wide; run without unrelated traffic.",
    }

    def safe_json(value, **kwargs):
        # Redact even if a misbehaving endpoint echoes the Authorization value.
        encoded = json.dumps(value, ensure_ascii=False, **kwargs)
        escaped_secret = json.dumps(secret, ensure_ascii=False)[1:-1]
        return encoded.replace(escaped_secret, "[REDACTED]") if secret else encoded

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(safe_json(report, indent=2) + "\n")

    def request(path, body=None, text=False, timeout=1800):
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["Authorization"] = "Bearer " + secret
        req = Request(args.base + path,
                      None if body is None else json.dumps(body).encode(), headers)
        with urlopen(req, timeout=timeout) as response:
            return response.read().decode() if text else json.load(response)

    def run(name, operation):
        row = {"name": name, "passed": False}
        start = time.monotonic()
        try:
            operation(row)
            row["passed"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, HTTPError):
                row["http_status"] = exc.code
                row["error_body"] = exc.read().decode(errors="replace")
        finally:
            row["elapsed_seconds"] = round(time.monotonic() - start, 3)
            report["checks"].append(row)
            save()
            print(safe_json(row), flush=True)
        return row

    def require(condition, explanation):
        if not condition:
            raise ValueError(explanation)

    def chat(messages, max_tokens=512, defaults=False, **extra):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if not defaults:
            body.update(temperature=0, chat_template_kwargs={"enable_thinking": False})
        body.update(extra)
        return request("/v1/chat/completions", body)

    def record_response(row, response):
        row["response"] = response
        row["usage"] = response.get("usage", {})
        row["prompt_tokens"] = row["usage"].get("prompt_tokens")
        row["output_tokens"] = row["usage"].get("completion_tokens")
        require(isinstance(row["prompt_tokens"], int) and row["prompt_tokens"] > 0,
                "Server omitted positive actual prompt token usage")
        require(isinstance(row["output_tokens"], int) and row["output_tokens"] > 0,
                "Server omitted positive actual completion token usage")
        choice = response["choices"][0]
        require(choice.get("finish_reason") != "length", "Output exhausted its token allowance")
        return choice.get("message", {}).get("content") or choice.get("text") or ""

    def exact_chat(row, messages, expected):
        row["expected"] = expected
        content = record_response(row, chat(messages))
        require(json_answer(content) == expected, "Final JSON did not exactly match planted facts")
        return content

    def metrics():
        raw = request("/metrics", text=True, timeout=30)
        samples = {}
        for line in raw.splitlines():
            match = re.match(r'((?:vllm:)?prefix_cache_(?:hits|queries)_total(?:\{[^}]*\})?)\s+(\S+)', line)
            if match:
                value = float(match[2])
                require(math.isfinite(value), "Nonfinite prefix cache metric")
                samples[match[1]] = value
        require(any("prefix_cache_hits_total" in name for name in samples),
                "Server did not expose prefix_cache_hits_total")
        return samples

    def tokenize(messages):
        result = request("/tokenize", {
            "model": model, "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        count = result.get("count")
        require(isinstance(count, int) and count > 0, "Invalid /tokenize count")
        tokens = result.get("tokens")
        if tokens is not None:
            require(isinstance(tokens, list) and len(tokens) == count
                    and all(type(token) is int and token >= 0 for token in tokens),
                    "/tokenize token IDs disagree with its count")
        return result

    def short_checks():
        def basic(row):
            content = record_response(row, chat([message("What is 1234 times 17? Reply with digits only.")]))
            require(content.strip() == "20978", "Arithmetic answer was not 20978")
        run("arithmetic", basic)

        def defaults(row):
            response = chat([message("What is the capital city of France? Answer in one short sentence.")],
                            max_tokens=8192, defaults=True)
            content = record_response(row, response)
            msg = response["choices"][0]["message"]
            fields = {key: msg[key] for key in ("reasoning", "reasoning_content") if key in msg}
            row["reasoning_fields_available"] = list(fields)
            row["request_overrides"] = {"max_tokens": 8192}
            require(bool(content.strip()) and "paris" in content.lower(), "Missing appropriate final answer")
            require("<think>" not in content and "</think>" not in content,
                    "Reasoning markup leaked into final answer")
            if fields:
                require(any(isinstance(value, str) and value.strip() for value in fields.values()),
                        "Reasoning field is present but empty under server defaults")
            else:
                row["reasoning_note"] = "No separate reasoning field exposed by endpoint"
        run("default-settings-medium-reasoning", defaults)

        def tools(row):
            nonce = uuid4().hex
            expected = {"city": "Paris", "units": "celsius", "nonce": nonce}
            schema = {"type": "function", "function": {
                "name": "preview_weather", "description": "Safe fictional weather preview; no side effects.",
                "parameters": {"type": "object", "additionalProperties": False,
                    "properties": {key: {"type": "string", "enum": [value]} for key, value in expected.items()},
                    "required": list(expected)},
            }}
            response = chat([message(f"Call preview_weather once for Paris in celsius using nonce {nonce}.")],
                            tools=[schema], tool_choice={"type": "function", "function": {"name": "preview_weather"}})
            record_response(row, response)
            calls = response["choices"][0]["message"].get("tool_calls") or []
            row["expected_arguments"] = expected
            row["tools_executed"] = False
            require(len(calls) == 1 and calls[0].get("type") == "function", "Expected exactly one function call")
            function = calls[0]["function"]
            require(function["name"] == "preview_weather", "Unexpected function requested")
            require(json.loads(function["arguments"]) == expected, "Tool arguments failed exact schema/value check")
        run("safe-tool-call-json", tools)

        def concurrency(row):
            tasks = [
                ("In Python, what is sum([3, 5, 8])?", "16"),
                ("In Python, what is sorted([9, 2, 7])[0]?", "2"),
                ("In Python, what is len('copper')?", "6"),
                ("Which planet is known as the Red Planet?", "Mars"),
                ("What is the chemical symbol of gold?", "Au"),
                ("How many sides does a hexagon have?", "6"),
            ]
            nonces = [uuid4().hex for _ in tasks]
            barrier = Barrier(len(tasks))
            def worker(i):
                item = {"name": f"concurrent-{i}", "nonce": nonces[i], "passed": False}
                start = time.monotonic()
                try:
                    question, answer = tasks[i]
                    prompt = (f"{question} Return only JSON with string fields answer and nonce. "
                              f"Use nonce {nonces[i]}. No explanation.")
                    barrier.wait(timeout=30)
                    response = chat([message(prompt)])
                    content = record_response(item, response)
                    require(json_answer(content) == {"answer": answer, "nonce": nonces[i]},
                            "Wrong factual answer or request nonce")
                    leaked = [nonce for nonce in nonces if nonce != nonces[i] and nonce in json.dumps(response)]
                    require(not leaked, "Another simultaneous request's nonce leaked into response")
                    item["passed"] = True
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, HTTPError):
                        item["http_status"] = exc.code
                        item["error_body"] = exc.read().decode(errors="replace")
                item["elapsed_seconds"] = round(time.monotonic() - start, 3)
                return item
            with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                row["requests"] = list(pool.map(worker, range(len(tasks))))
            require(all(item["passed"] for item in row["requests"]), "Concurrent factual isolation check failed")
        run("concurrent-code-prose-isolation", concurrency)

        def prefix(row):
            values = {key: f"{key}-{uuid4().hex[:12]}" for key in ("archive", "maintenance", "inventory")}
            records = [f"Record {i:05d}: copper paper window marble." for i in range(4000)]
            for index, (key, value) in zip((120, 2000, 3880), values.items()):
                records[index] = f"The {key} access phrase is {value}."
            messages = [message("Remember these records.\n" + "\n".join(records)
                                + "\nReturn only a JSON object with archive, maintenance, inventory phrases.")]
            row["turns"] = []
            for turn in range(12):
                item = {"turn": turn + 1, "expected": dict(values)}
                row["turns"].append(item)
                start = time.monotonic()
                content = exact_chat(item, messages, dict(values))
                item["elapsed_seconds"] = round(time.monotonic() - start, 3)
                if turn:
                    require(item["prompt_tokens"] > row["turns"][turn - 1]["prompt_tokens"],
                            "Shared-prefix conversation did not grow in actual input tokens")
                item["passed"] = True
                messages.append({"role": "assistant", "content": content})
                if turn == 0:
                    row["metrics_before_followups"] = metrics()
                key = ("maintenance", "archive", "inventory")[turn % 3]
                values[key] = f"{key}-{uuid4().hex[:12]}"
                messages.append(message(f"Replace only the {key} phrase with {values[key]}. "
                                        "Recall the other two from the records. Return only the complete JSON object."))
            before = row["metrics_before_followups"]
            deadline = time.monotonic() + 15
            while True:
                after = metrics()
                row["metrics_after_followups"] = after
                require(all(key in after and after[key] >= value for key, value in before.items()),
                        "Prefix metric series disappeared or reset during check")
                delta = sum(after[key] - value for key, value in before.items() if "prefix_cache_hits_total" in key)
                row["prefix_cache_hits_delta"] = delta
                if delta > 0 or time.monotonic() >= deadline:
                    break
                time.sleep(1)
            require(delta > 0, "Growing conversation did not produce measured prefix-cache reuse")
        run("growing-prefix-exact-recall-and-reuse", prefix)

    def long_checks():
        state = {}
        def build(row):
            values = {key: f"{key}-{uuid4().hex[:12]}" for key in ("archive", "maintenance", "inventory")}
            count = 14000
            row["tokenization_attempts"] = []
            for _ in range(10):
                records = [f"Record {i:06d}: copper paper window marble river cloud." for i in range(count)]
                positions = [int(count * fraction) for fraction in (0.03, 0.50, 0.97)]
                for index, (key, value) in zip(positions, values.items()):
                    records[index] = f"The {key} access phrase is {value}."
                messages = [message("Remember the three labeled access phrases in these records.\n"
                                    + "\n".join(records)
                                    + "\nReturn only a JSON object with keys archive, maintenance, inventory and their exact phrases.")]
                encoded = tokenize(messages)
                actual = encoded["count"]
                row["tokenization_attempts"].append({"records": count, "prompt_tokens": actual})
                if 260000 <= actual <= CONTEXT - LONG_OUTPUT:
                    break
                count = max(3, round(count * 260500 / actual))
            require(260000 <= actual <= CONTEXT - LONG_OUTPUT, "Could not construct near-full tokenized context")
            row.update(prompt_tokens=actual, expected=values, needle_record_positions=positions,
                       max_output_tokens=LONG_OUTPUT, context_limit=CONTEXT)
            state.update(messages=messages, encoded=encoded, expected=values, records=records)
        built = run("near-full-prompt-tokenization", build)
        if not built["passed"]:
            return

        def probe(row):
            encoded = tokenize([message("Reply with the single word READY.")])
            if encoded.get("tokens") is None:
                row["supported"] = False
                row["reason"] = "/tokenize does not return token IDs; using measured chat prompt"
                return
            try:
                response = request("/v1/completions", {
                    "model": model, "prompt": encoded["tokens"], "max_tokens": 128, "temperature": 0,
                })
            except HTTPError as exc:
                if exc.code not in (400, 404, 405, 422):
                    raise
                row.update(supported=False, http_status=exc.code,
                           error_body=exc.read().decode(errors="replace"),
                           reason="Token-ID completion probe rejected; using measured chat prompt")
                return
            record_response(row, response)
            require(row["prompt_tokens"] == encoded["count"], "Completion altered supplied token-ID count")
            state["token_ids"] = True
            row["supported"] = True
        run("token-id-completions-capability", probe)

        def retrieval(row):
            row.update(tokenized_prompt_tokens=state["encoded"]["count"], expected=state["expected"])
            row["metrics_before"] = metrics()
            if state.get("token_ids"):
                row["endpoint"] = "/v1/completions"
                response = request(row["endpoint"], {
                    "model": model, "prompt": state["encoded"]["tokens"],
                    "temperature": 0, "max_tokens": LONG_OUTPUT,
                })
            else:
                row["endpoint"] = "/v1/chat/completions"
                response = chat(state["messages"], max_tokens=LONG_OUTPUT)
            content = record_response(row, response)
            row["metrics_after"] = metrics()
            require(row["prompt_tokens"] == state["encoded"]["count"], "Usage count differs from /tokenize; possible truncation/template mismatch")
            require(260000 <= row["prompt_tokens"] <= CONTEXT - LONG_OUTPUT, "Actual input was not near-full context")
            require(json_answer(content) == state["expected"], "Near-full context needle retrieval failed")
        run("near-full-three-needle-retrieval", retrieval)

        def over_limit(row):
            if state.get("token_ids"):
                tokens = state["encoded"]["tokens"]
                prompt = tokens + [tokens[len(tokens) // 2]] * (CONTEXT + 1 - len(tokens))
                row.update(prompt_tokens=len(prompt), endpoint="/v1/completions")
                body = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}
            else:
                text = state["messages"][0]["content"] + "\n" + "\n".join(state["records"][:1000])
                messages = [message(text)]
                encoded = tokenize(messages)
                row.update(prompt_tokens=encoded["count"], endpoint="/v1/chat/completions")
                body = {"model": model, "messages": messages, "max_tokens": 1,
                        "chat_template_kwargs": {"enable_thinking": False}}
            require(row["prompt_tokens"] > CONTEXT, "Over-limit probe did not exceed actual context")
            try:
                response = request(row["endpoint"], body)
            except HTTPError as exc:
                error = exc.read().decode(errors="replace")
                row.update(http_status=exc.code, error_body=error)
                require(exc.code == 400 and re.search(r"context|maximum.*(?:length|tokens)|too (?:long|many tokens)", error, re.I),
                        "Rejection was not a context-limit validation error")
            else:
                row["response"] = response
                raise ValueError("Server accepted an over-limit prompt (possibly silently truncated)")
        run("over-native-limit-rejection", over_limit)

    try:
        parsed = urlsplit(args.base)
        require(parsed.scheme in ("http", "https") and parsed.netloc
                and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
                "--base must be an HTTP(S) endpoint without credentials, query or fragment")
        if args.api_key_file:
            secret = args.api_key_file.read_text().strip()
            require(bool(secret) and "\n" not in secret and "\r" not in secret, "API key file must contain one nonempty line")
        request("/health", text=True, timeout=30)
        def metadata(row):
            response = request("/v1/models", timeout=30)
            row["models"] = response
            candidates = [item for item in response["data"] if item.get("id") == model]
            require(len(candidates) == 1, f"Expected served model {model}")
            row["max_model_len"] = candidates[0].get("max_model_len")
            require(row["max_model_len"] == CONTEXT, "Model metadata does not advertise native 262144 context")
        run("model-native-context-metadata", metadata)
        if args.mode in ("short", "all"):
            short_checks()
        if args.mode in ("long", "all"):
            long_checks()
        report["passed"] = bool(report["checks"]) and all(row["passed"] for row in report["checks"])
    except BaseException as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, HTTPError):
            report["http_status"] = exc.code
            report["error_body"] = exc.read().decode(errors="replace")
    finally:
        save()
    if not report["passed"]:
        raise SystemExit("Serving acceptance failed; see JSON evidence")


if __name__ == "__main__":
    main()
