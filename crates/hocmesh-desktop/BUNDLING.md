# Bundling hocMESH Desktop

Every hocMESH install is a whole peer. There is no client and no server: the
same machine serves -- lending CPU, memory and GPU to other people's work --
and spends what that earns on work of its own, the way a torrent client seeds
the file it just finished downloading. hocMESH only runs that trade in the
other order: you seed first, and what you earn by seeding is what lets you
later reach for somebody else's hardware.

So an installer never carries "the client half". It carries the node, the
coordinator and the validator, and the desktop installer adds a window over
them. The two installers differ in exactly one thing -- whether the machine
has a screen -- which is why they **replace** each other rather than sitting
side by side.

The window is not the node. `hocmesh-desktop` starts, watches and stops
`hocmesh daemon`, a separate executable that keeps running when the window is
closed -- the same split Docker Desktop draws between its window and its
engine. An installer therefore has to lay the binaries down side by side, so
that `supervisor::discover_node` finds the node next to the app rather than
falling back to whatever older build is on the operator's `PATH`.

Tauri calls those extra executables *sidecars*, and declares them with
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
as their first argument -- passed in rather than built here so the window and
the daemons it starts always come from one build -- find `hocmesh-coordinator`
and `hocmesh-validator` beside it, stage all three as
`binaries/<name>-<target-triple>`, run the bundler with that patch, then copy
each installer into `dist/` under a name that carries the version. Each script
then opens what it produced and fails unless all four executables are inside:
an installer carrying the window without the peer would look perfectly fine
from the outside and install an app that cannot start anything. Tauri strips
the triple when it bundles, so the node is installed as plain `hocmesh` (or
`hocmesh.exe`) next to the app binary, which is exactly the first place
`candidate_paths` looks.

## Why the two Linux packages replace each other

Tauri's Debian bundler copies the app binary and every sidecar into `/usr/bin`,
so the desktop `.deb` owns `/usr/bin/hocmesh` -- and so does the headless one.
dpkg refuses to let two packages own one path:

```
dpkg: error processing archive hoc-mesh-desktop_0.3.0_amd64.deb
 trying to overwrite '/usr/bin/hocmesh', which is also in package hocmesh
```

That is not a collision to work around by renaming a binary. Both packages
install the same peer, and a machine wants one of them, so they say so:
`tauri.bundle.json` gives the desktop package `Provides`, `Conflicts` and
`Replaces: hocmesh`, and `packaging/linux/control.in` gives the headless
package `Conflicts` and `Replaces: hoc-mesh-desktop`. `apt install` either one
on a machine that has the other and dpkg swaps them cleanly, leaving
`/usr/bin/hocmesh` as the command an operator types in both cases.
`package-desktop.sh` reads those three fields back out of the built `.deb` with
`dpkg-deb --field`, so an edit that silently dropped them fails the build
rather than shipping a package that refuses to install.

## What comes out

| Platform | Artifacts |
| --- | --- |
| Windows | `.msi` (WiX) and `.exe` (NSIS, per-machine) |
| macOS | `.dmg` |
| Linux | `.deb` and `.AppImage` |

Each carries the whole peer plus the window. The headless installers that
`package-windows.ps1`, `package-macos.sh` and `package-linux.sh` produce carry
the same three command line binaries and no window; install those on a machine
with no screen.

## Prerequisites

* `cargo tauri` -- `cargo install tauri-cli --version 2.11.4 --locked`.
  The packaging scripts install it themselves if it is missing.
* A release build of all three peer binaries, since the scripts stage what
  they are handed rather than building it:
  `cargo build --release -p hocmesh -p hocmesh-coordinator -p hocmesh-validator`.
* Linux only: `libwebkit2gtk-4.1-dev`, `libjavascriptcoregtk-4.1-dev`,
  `libsoup-3.0-dev`, `libappindicator3-dev`, `librsvg2-dev`, `patchelf`.
  Build the AppImage on a host that has FUSE: `linuxdeploy` is itself an
  AppImage, which is why CI pins that job to `ubuntu-22.04`.
* Windows and macOS need nothing extra; the bundler fetches its own WiX and
  NSIS toolchains on first use.
