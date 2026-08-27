"""Cheap structural checks that need no toolchain.

Run this when `cargo` is not available or not worth the wait. It is not a
substitute for `cargo check`/`test`/`clippy`: everything here is either a parse
of a data file or a brace balance, and a file can be balanced and still be
nonsense.
"""

from pathlib import Path
import json
import sys
import tomllib

root = Path(__file__).resolve().parents[1]

# Build output is not source. It holds thousands of generated JSON files and a
# handful of generated `.rs` files that this repository did not write and is
# not answerable for, and walking it makes a "cheap" check take seconds.
SKIP = {"target", ".git"}


def sources(pattern):
    for path in sorted(root.rglob(pattern)):
        if not SKIP.intersection(path.relative_to(root).parts):
            yield path


def code_braces(text):
    """Yield ``(index, brace)`` for every brace that is Rust code.

    Braces inside comments, string literals and character literals are not
    structure -- a test fixture containing the string ``"{ not json"`` is
    perfectly balanced code -- so counting them reports a mismatch that is not
    there. This walks the tokens well enough to tell the difference: line and
    (nestable) block comments, byte/raw/ordinary strings, and character
    literals as distinct from lifetimes.
    """
    i, n = 0, len(text)
    ident = lambda c: c.isalnum() or c == "_"
    while i < n:
        c = text[i]

        # -- comments -------------------------------------------------------
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            i = text.find("\n", i)
            if i == -1:
                return
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            continue

        # -- raw strings: r"..", r#".."#, br##".."## -------------------------
        if c in "rb" and (i == 0 or not ident(text[i - 1])):
            j = i + 1 if c == "b" and i + 1 < n and text[i + 1] == "r" else i
            if text[j] == "r":
                k = j + 1
                while k < n and text[k] == "#":
                    k += 1
                if k < n and text[k] == '"':
                    fence = '"' + "#" * (k - j - 1)
                    end = text.find(fence, k + 1)
                    i = n if end == -1 else end + len(fence)
                    continue

        # -- ordinary and byte strings --------------------------------------
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue

        # -- character literals, which `'` also starts a lifetime with -------
        if c == "'":
            if i + 1 < n and text[i + 1] == "\\":
                # An escape runs to the closing quote. `'\u{1F600}'` is why the
                # braces in here must not be counted.
                i += 2
                while i < n and text[i] != "'":
                    i += 1
                i += 1
            elif i + 2 < n and text[i + 2] == "'":
                i += 3  # 'x', including '{' and '}'
            else:
                i += 1  # a lifetime or a loop label: 'a, 'static, 'outer
            continue

        if c in "{}":
            yield i, c
        i += 1


def check_braces(path):
    text = path.read_text(encoding="utf-8")
    where = lambda index: text.count("\n", 0, index) + 1
    stack = []
    for index, brace in code_braces(text):
        if brace == "{":
            stack.append(index)
        elif stack:
            stack.pop()
        else:
            return f"{path.relative_to(root)}:{where(index)}: `}}` closes nothing"
    if stack:
        return f"{path.relative_to(root)}:{where(stack[-1])}: `{{` is never closed"
    return None


def self_test():
    """The scanner has to skip enough, and no more.

    A scanner that skipped everything would report a clean tree forever, so
    prove it still sees real code and still refuses these decoys.
    """
    decoys = [
        'fn f() { let s = "{ not json"; }',
        "fn f() { let c = '{'; }",
        "fn f() { /* } */ }",
        "fn f() { /* /* } */ */ }",
        'fn f() { let s = r#"} { "#; }',
        'fn f() { let s = br"}"; }',
        "fn f() { let e = '\\u{7d}'; }",
        "fn f() { let s = \"\\\"}\"; }",
        "fn f<'a>(x: &'a str) { let _ = x; }",
        "fn f() { 'outer: loop { break 'outer; } }",
        "// }\nfn f() {}",
        '/// ```\n/// let x = "{";\n/// ```\nfn f() {}',
    ]
    for source in decoys:
        braces = [b for _, b in code_braces(source)]
        assert braces.count("{") == braces.count("}"), source
        assert braces, f"scanner went blind on: {source}"
    for broken, expect in [("fn f() {", "{"), ("fn f() }", "}")]:
        braces = [b for _, b in code_braces(broken)]
        assert braces == [expect], broken


self_test()

for path in sources("*.toml"):
    tomllib.loads(path.read_text(encoding="utf-8"))
for path in sources("*.json"):
    json.loads(path.read_text(encoding="utf-8"))

workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
for member in workspace["workspace"]["members"]:
    assert (root / member / "Cargo.toml").is_file(), member

problems = [problem for problem in map(check_braces, sources("*.rs")) if problem]
if problems:
    print("\n".join(problems), file=sys.stderr)
    raise SystemExit(1)

print("Static repository checks passed. This is not a substitute for cargo check/test/clippy.")
