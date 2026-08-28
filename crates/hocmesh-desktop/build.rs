fn main() {
    // Without `gui` there is no webview to embed a frontend into and no
    // bundler to configure, so the build script has nothing to do -- which is
    // what lets the rest of the crate compile where Tauri's system
    // dependencies are not installed.
    #[cfg(feature = "gui")]
    tauri_build::build();
}
