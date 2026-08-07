"""Convert a conversation dataset into KVault shards the training loader can index.

Values are JSON ``{"messages": [{"role", "content"}, ...]}`` rather than the
UTF-8 document text the plain corpus stores, so read them with
``CorpusRecords(..., schema="json")`` and render them with ``chat``.
Keys are sequential big-endian int64, as everywhere else in the vault.

    python scripts/data/to_vault_chat.py --src /Iolite/text-dataset/sft/smoltalk \\
        --name sft/smoltalk
    python scripts/data/to_vault_chat.py --src ... --name ... --dry-run

See internal/corpus-data-setup.md.
"""

import argparse
import fnmatch
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

VAULT = "/Iolite/text-dataset/_vault"
STAGE = "/xg7/vault-stage"
FILES_PER_SHARD = 3
MIN_REPLY_CHARS = 8
ROLES = ("system", "user", "assistant", "tool")
MESSAGE_FIELDS = ("messages", "conversations", "conversation")
PROMPT_FIELDS = ("instruction", "prompt", "question", "query", "input")
REPLY_FIELDS = ("response", "output", "completion", "answer")
SYSTEM_FIELDS = ("system", "system_prompt", "system_message")
TOOL_FIELDS = ("tools", "functions", "tool_schema")
CALL_FIELDS = ("function_calls", "tool_calls")
# Reserved at 64021-64024. See internal/chat-template-design.md section 4.
TOOL_OPEN, TOOL_CLOSE = "<|tool_call|>", "<|/tool_call|>"
ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "environment": "tool",
    "observation": "tool",
    "function": "tool",
}
# Per-worker budgets, as in _meta/to_vault.py: both multiply by the worker count.
BATCH_ROWS = 200_000
BATCH_BYTES = 512 << 20
CACHE_KB = 256_000
MMAP_BYTES = 1 << 30


READABLE = (".parquet", ".jsonl", ".json")


def input_files(src: str, pattern: str = "*") -> list[str]:
    """Every readable table under ``src`` matching ``pattern``, minus the test split.

    Reads parquet, JSON Lines, and a JSON file holding a list or ``{"data": [...]}``.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if (
                name.endswith(READABLE)
                and not name.startswith("test-")
                and fnmatch.fnmatch(name, pattern)
            ):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def first_of(row: dict, fields: tuple[str, ...]):
    """The first truthy value among ``fields``, or ``""``."""
    for field in fields:
        value = row.get(field)
        if value:
            return value
    return ""


def as_text(value) -> str:
    """A declaration or call payload as text, whether it arrives as JSON or a string."""
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def as_messages(row: dict) -> list[dict]:
    """One parquet row -> ``[{"role", "content"}]``, or ``[]`` if unusable.

    Accepts a message list under any of :data:`MESSAGE_FIELDS`, or a flat
    prompt / reply pair. Blank turns are dropped; an unknown role becomes
    ``user``. A tool declaration beside the list, or on a message, is folded
    into the system turn; a call is wrapped in ``<|tool_call|>``.
    See internal/chat-template-design.md.
    """
    raw = None
    for field in MESSAGE_FIELDS:
        value = row.get(field)
        # Some sets store the message list as a JSON string in a string column.
        if isinstance(value, str) and value.lstrip().startswith("["):
            try:
                value = json.loads(value)
            except ValueError:
                value = None
        if isinstance(value, (list, tuple)) and value:
            raw = value
            break
    from_list = raw is not None
    if raw is None:
        prompt = first_of(row, PROMPT_FIELDS)
        reply = first_of(row, REPLY_FIELDS)
        system = first_of(row, SYSTEM_FIELDS)
        raw = []
        if system:
            raw.append({"role": "system", "content": system})
        if prompt:
            raw.append({"role": "user", "content": prompt})
        if reply:
            raw.append({"role": "assistant", "content": reply})

    messages = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        content = (message.get("content") or message.get("value") or "").strip()
        role = (message.get("role") or message.get("from") or "user").lower()
        role = ROLE_ALIASES.get(role, role)
        role = role if role in ROLES else "user"
        # A declaration rides on the system turn, a call on the assistant's.
        decl = as_text(first_of(message, TOOL_FIELDS))
        if decl and role == "system":
            content = f"{content}\n{decl}".strip()
        call = as_text(first_of(message, CALL_FIELDS))
        if call:
            content = f"{content}\n{TOOL_OPEN}{call}{TOOL_CLOSE}".strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})

    if from_list:
        head = "\n".join(
            x
            for x in (
                str(first_of(row, SYSTEM_FIELDS)).strip(),
                as_text(first_of(row, TOOL_FIELDS)),
            )
            if x
        )
    else:
        head = ""
    if head and messages:
        if messages[0]["role"] == "system":
            messages[0]["content"] = f"{head}\n{messages[0]['content']}".strip()
        else:
            messages.insert(0, {"role": "system", "content": head})
    return messages


def usable(messages: list[dict]) -> bool:
    """A conversation is usable when a non-trivial reply follows some context."""
    for i, message in enumerate(messages):
        if (
            message["role"] == "assistant"
            and i > 0
            and len(message["content"]) >= MIN_REPLY_CHARS
        ):
            return True
    return False


def flagged(row: dict, flags: tuple[str, ...]) -> bool:
    """Whether the row carries a truthy value in any of the named columns."""
    return any(row.get(f) for f in flags)


def dedup_key(messages: list[dict], mode: str) -> str:
    """Fingerprint for duplicate detection.

    ``prompt`` keys on the first user message, ``conversation`` on the whole
    exchange. Keys are held per shard job. See internal/corpus-data-setup.md.
    """
    if mode == "prompt":
        text = next((m["content"] for m in messages if m["role"] == "user"), "")
    else:
        text = "\n".join(m["content"] for m in messages)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def rows_parquet(path: str):
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    for group in range(handle.metadata.num_row_groups):
        for row in handle.read_row_group(group).to_pylist():
            yield row


def rows_jsonl(path: str):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def rows_json(path: str):
    with open(path, encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    items = payload if isinstance(payload, list) else payload.get("data", [])
    for row in items:
        if isinstance(row, dict):
            yield row


def rows_of(path: str):
    """Dispatch to the reader for this file's extension."""
    if path.endswith(".parquet"):
        return rows_parquet(path)
    if path.endswith(".jsonl"):
        return rows_jsonl(path)
    return rows_json(path)


def convert(job) -> tuple[int, int, int, bool]:
    """Write one vault shard from a list of input files."""
    import kohakuvault as kv

    name, files, index, vault_dir, stage_dir, drop_flags, dedup_on = job
    stem = name.replace("/", "__")
    final = os.path.join(vault_dir, f"{stem}.{index:04d}.db")
    marker = final + ".done"
    if os.path.exists(marker):
        with open(marker) as handle:
            return index, int(handle.read().strip() or 0), 0, True

    os.makedirs(stage_dir, exist_ok=True)
    out = os.path.join(stage_dir, f"{stem}.{index:04d}.db")
    for stale in (out, out + "-wal", out + "-shm"):
        if os.path.exists(stale):
            os.remove(stale)

    vault = kv.KVault(out, page_size=65536, cache_kb=CACHE_KB, mmap_size=MMAP_BYTES)
    vault.enable_auto_pack()
    count = skipped = dropped = duped = pending = 0
    seen: set[str] = set()
    batch = []
    began = time.time()
    for file_no, path in enumerate(files, start=1):
        for row in rows_of(path):
            if drop_flags and flagged(row, drop_flags):
                dropped += 1
                continue
            messages = as_messages(row)
            if not usable(messages):
                skipped += 1
                continue
            if dedup_on:
                key = dedup_key(messages, dedup_on)
                if key in seen:
                    duped += 1
                    continue
                seen.add(key)
            record = {"messages": messages}
            if row.get("source"):
                record["subset"] = row["source"]
            blob = json.dumps(record, ensure_ascii=False).encode("utf-8")
            batch.append((count.to_bytes(8, "big"), blob))
            pending += len(blob)
            count += 1
            if len(batch) >= BATCH_ROWS or pending >= BATCH_BYTES:
                vault.update(iter(batch))
                batch.clear()
                pending = 0
        elapsed = time.time() - began
        print(
            f"  PROGRESS {stem}.{index:04d} file {file_no}/{len(files)} "
            f"{count:>10,} rows {skipped:>8,} unusable {dropped:>8,} flagged "
            f"{duped:>9,} dup {elapsed:6.0f}s "
            f"{count / max(elapsed, 1):>7,.0f} rows/s",
            flush=True,
        )
    if batch:
        vault.update(iter(batch))
    vault.close()
    for tail in ("-wal", "-shm"):
        if os.path.exists(out + tail):
            os.remove(out + tail)
    shutil.move(out, final)
    with open(marker, "w") as handle:
        handle.write(str(count))
    return index, count, skipped + dropped + duped, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of parquet files")
    ap.add_argument("--name", required=True, help='"<category>/<dataset>"')
    ap.add_argument("--vault", default=VAULT)
    ap.add_argument("--stage", default=STAGE)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--files-per-shard", type=int, default=FILES_PER_SHARD)
    ap.add_argument(
        "--drop-if",
        nargs="*",
        default=[],
        metavar="COLUMN",
        help="skip a row when any of these columns is truthy, e.g. toxic redacted",
    )
    ap.add_argument(
        "--dedup-on",
        choices=("prompt", "conversation"),
        default=None,
        help="keep one row per prompt, or per whole conversation",
    )
    ap.add_argument(
        "--pattern",
        default="*",
        help="only convert parquet basenames matching this glob, e.g. 'tool-*'",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = input_files(args.src, args.pattern)
    if not files:
        print(f"no parquet/jsonl/json under {args.src}")
        return 1
    jobs = [
        (
            args.name,
            files[i : i + args.files_per_shard],
            i // args.files_per_shard,
            args.vault,
            args.stage,
            tuple(args.drop_if),
            args.dedup_on,
        )
        for i in range(0, len(files), args.files_per_shard)
    ]
    print(
        f"{args.name}: {len(files)} files -> {len(jobs)} shards"
        f"{', dropping rows flagged ' + '/'.join(args.drop_if) if args.drop_if else ''}",
        flush=True,
    )
    if args.dry_run:
        for _, group, index, *_ in jobs:
            print(
                f"  {index:04d}: {len(group)} files, first {os.path.basename(group[0])}"
            )
        return 0

    os.makedirs(args.vault, exist_ok=True)
    total = dropped = 0
    began = time.time()
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for index, count, skipped, cached in pool.imap_unordered(convert, jobs):
            total += count
            dropped += skipped
            state = "cached" if cached else "written"
            print(f"  shard {index:04d} {state}: {count:,} rows", flush=True)
    print(
        f"\n{args.name}: {total:,} rows, {dropped:,} unusable, "
        f"{time.time() - began:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
