#!/usr/bin/env bash
# regression test for llm-d#2478 — Redis Pub/Sub must be marked deprecated
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "PASS: $*"; }
grep -q "Redis Pub/Sub (deprecated)" "$ROOT/docs/architecture/advanced/batch/async-processor.md" || fail "arch doc missing 'Redis Pub/Sub (deprecated)' row"
ok "arch deprecated row present"
grep -Fq 'Deprecation Notice' "$ROOT/docs/architecture/advanced/batch/async-processor.md" || fail "arch doc missing Deprecation Notice"
grep -q "llm-d-async#417" "$ROOT/docs/architecture/advanced/batch/async-processor.md" || fail "arch doc missing llm-d-async#417 link"
grep -Fq '[!WARNING]' "$ROOT/docs/architecture/advanced/batch/async-processor.md" || fail "arch doc missing [!WARNING] admonition"
ok "arch warning callout + link present"
grep -q "Redis Pub/Sub (deprecated)" "$ROOT/docs/operations/async-processor.md" || fail "ops doc missing deprecated qualifier"
grep -Fq 'Message Queue Integrations' "$ROOT/docs/operations/async-processor.md" || fail "ops doc missing cross-ref to Message Queue Integrations"
ok "ops deprecated qualifier + cross-ref present"
if grep -q "Redis Pub/Sub fans out" "$ROOT/docs/operations/async-processor.md"; then fail "ops doc still has bare 'Redis Pub/Sub fans out'"; fi
ok "ops bare fan-out absent"
if grep -n "Redis Pub/Sub" "$ROOT/docs/architecture/advanced/batch/async-processor.md" "$ROOT/docs/operations/async-processor.md" | grep -v -i "deprecat" >/dev/null 2>&1; then fail "bare Redis Pub/Sub hit remains (without deprecated qualifier)"; fi
ok "no bare Redis Pub/Sub hit"
echo "all checks passed for #2478"
