#!/usr/bin/env python3
"""Stop this deployment before unified-memory exhaustion can hang the Spark."""
import json
from pathlib import Path
import subprocess
import time

NAME = "qwen38-flash"
MIN_AVAILABLE_KIB = 6 * 1024 * 1024

while True:
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(memory["MemAvailable"].split()[0]) < MIN_AVAILABLE_KIB:
        result = subprocess.run(["docker", "inspect", NAME], capture_output=True, text=True)
        if result.returncode == 0:
            container = json.loads(result.stdout)[0]
            if (container["State"]["Running"] and
                    container["Config"].get("Labels", {}).get("local-ai.managed") == "true"):
                print("Stopping local AI: MemAvailable below 6 GiB", flush=True)
                subprocess.run(["docker", "stop", "--time", "10", NAME], check=True, timeout=30)
                # An intentional stop prevents Docker's restart policy from causing an OOM loop.
    time.sleep(5)
