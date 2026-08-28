# Bundling hocMESH Desktop

The window is not the node. `hocmesh-desktop` starts, watches and stops
`hocmesh daemon`, which is a separate executable that keeps running when the
window is closed -- the same split Docker Desktop draws between its window and
its engine. An installer therefore has to lay *both* binaries down, side by
side, so that `supervisor::discover_node` finds the node next to the app rather
than falling back to whatever older build is on the operator's `PATH`.

Tauri calls that second executable a *sidecar*, and declares it with
`bundle.externalBin`. It is deliberately **not** in `tauri.conf.json`:
`tauri-build` verifies at compile time that every declared sidecar exists for
the host target triple, so putting it in the base config would make
`cargo build`, `cargo test` and `cargo clippy` fail on a clean checkout until
somebody had staged a release build of another crate. The declaration lives in
`tauri.bundle.json` instead, a config patch that only the packaging scripts
pass:

```
cargo tauri build --config tauri.bundle.json -- --locked
```

`scripts/package-desktop.ps1` (Windows) and `scripts/package-desktop.sh`
(macOS, Linux) do the whole sequence: take an already-built release `hocmesh`
as their first argument -- passed in rather than built here so the app and the
daemon it will start always come from one build -- stage it as
`binaries/hocmesh-node-<target-triple>`, run the bundler with that patch, then
copy each installer into `dist/` under a name that carries the version. Each
script then opens what it produced and fails unless both executables are inside:
an installer carrying the window without the daemon would look perfectly fine
from the outside and install an app that cannot start anything. Tauri strips
the triple when it bundles, so the sidecar is installed as plain `hocmesh-node`
(or `hocmesh-node.exe`) next to the app binary, which is exactly the first place
`candidate_paths` looks.

## Why the sidecar is not called `hocmesh`

Tauri's Debian bundler copies both the app binary and every sidecar into
`/usr/bin`. The client package `hocmesh-compute-client` already owns
`/usr/bin/hocmesh`, and dpkg refuses to let two packages own one path, so a
sidecar named `hocmesh` would make the two installers mutually exclusive:

```
dpkg: error processing archive hocmesh-desktop_0.3.0_amd64.deb
 trying to overwrite '/usr/bin/hocmesh', which is also in package
 hocmesh-compute-client
```

The name is `hocmesh-node` on every platform so there is one answer rather than
a per-platform one, and `package-desktop.sh` fails the build if the `.deb` ever
claims `/usr/bin/hocmesh` again. `supervisor::NODE_BINARIES` looks for
`hocmesh-node` first and plain `hocmesh` second, so a machine that has only the
client package installed still gets a working daemon from the operator's
`PATH`, and a machine that has both is driven by the node this window shipped
with.

## What comes out

| Platform | Artifacts |
| --- | --- |
| Windows | `.msi` (WiX) and `.exe` (NSIS, per-machine) |
| macOS | `.dmg` |
| Linux | `.deb` and `.AppImage` |

These are the desktop app. They are separate from the client installers that
`package-windows.ps1`, `package-macos.sh` and `package-linux.sh` produce, which
carry the three command line binaries and no window.

## Prerequisites

* `cargo tauri` -- `cargo install tauri-cli --version 2.11.4 --locked`.
  The packaging scripts install it themselves if it is missing.
* Linux only: `libwebkit2gtk-4.1-dev`, `libjavascriptcoregtk-4.1-dev`,
  `libsoup-3.0-dev`, `libappindicator3-dev`, `librsvg2-dev`, `patchelf`.
  Build the AppImage on a host that has FUSE: `linuxdeploy` is itself an
  AppImage, which is why CI pins that job to `ubuntu-22.04`.
* Windows and macOS need nothing extra; the bundler fetches its own WiX and
  NSIS toolchains on first use.
