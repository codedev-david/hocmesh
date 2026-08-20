from pathlib import Path
import json, tomllib
root = Path(__file__).resolve().parents[1]
for p in root.rglob("*.toml"):
    tomllib.loads(p.read_text(encoding="utf-8"))
for p in root.rglob("*.json"):
    json.loads(p.read_text(encoding="utf-8"))
workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
for member in workspace["workspace"]["members"]:
    assert (root / member / "Cargo.toml").is_file(), member
for p in root.rglob("*.rs"):
    text = p.read_text(encoding="utf-8")
    assert text.count("{") == text.count("}"), f"gross brace mismatch: {p}"
print("Static repository checks passed. This is not a substitute for cargo check/test/clippy.")
