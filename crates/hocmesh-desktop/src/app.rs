//! The Tauri application: window, tray, commands, refresh loop.
//!
//! Everything in this module is glue. The rules it enforces were decided in
//! [`crate::dashboard`], [`crate::supervisor`] and [`crate::tray`], where they
//! can be tested without a display server; what is here is the part that needs
//! a real event loop, and it is kept thin on purpose so that little of the
//! app's behaviour lives where no test can reach it.

use crate::dashboard::Snapshot;
use crate::node::Node;
use crate::settings::Settings;
use crate::supervisor::{RunState, start_blocker};
use crate::tray::{self, TrayModel};
use hocmesh::control::{LimitsRequest, LimitsUpdate};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tauri::image::Image;
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, RunEvent, WindowEvent};
use tokio::sync::Mutex;

const TRAY_ID: &str = "hocmesh-tray";
const WINDOW_ID: &str = "main";
/// How often the tray and any open window refresh.
///
/// The daemon heartbeats every ten seconds, so polling much faster than this
/// would show the same numbers repeatedly while keeping a radio awake on a
/// laptop. Three seconds is fast enough that a Start click feels immediate.
const REFRESH: Duration = Duration::from_secs(3);

const ICON_WORKING: &[u8] = include_bytes!("../icons/tray-working.png");
const ICON_DEGRADED: &[u8] = include_bytes!("../icons/tray-degraded.png");
const ICON_STOPPED: &[u8] = include_bytes!("../icons/tray-stopped.png");

fn icon_bytes(name: &str) -> &'static [u8] {
    match name {
        "tray-working.png" => ICON_WORKING,
        "tray-degraded.png" => ICON_DEGRADED,
        _ => ICON_STOPPED,
    }
}

/// Everything the commands share.
pub struct Shared {
    node: Mutex<Node>,
    config_dir: PathBuf,
}

impl Shared {
    pub fn new(config_dir: PathBuf) -> Self {
        let settings = Settings::load(&config_dir).normalised();
        Self {
            node: Mutex::new(Node::new(settings)),
            config_dir,
        }
    }
}

type Shell = Arc<Shared>;

#[tauri::command]
async fn snapshot(state: tauri::State<'_, Shell>, before: Option<u64>) -> Result<Snapshot, String> {
    let mut node = state.node.lock().await;
    Ok(node.snapshot(before).await)
}

#[tauri::command]
async fn get_settings(state: tauri::State<'_, Shell>) -> Result<Settings, String> {
    let node = state.node.lock().await;
    Ok(node.settings().clone())
}

#[tauri::command]
async fn save_settings(
    state: tauri::State<'_, Shell>,
    settings: Settings,
) -> Result<Settings, String> {
    let settings = settings.normalised();
    settings.save(&state.config_dir).map_err(display)?;
    let mut node = state.node.lock().await;
    node.set_settings(settings.clone());
    Ok(settings)
}

#[tauri::command]
async fn start_node(state: tauri::State<'_, Shell>) -> Result<RunState, String> {
    let mut node = state.node.lock().await;
    node.start().map_err(display)
}

#[tauri::command]
async fn stop_node(state: tauri::State<'_, Shell>) -> Result<bool, String> {
    let mut node = state.node.lock().await;
    node.stop().await.map_err(display)
}

#[tauri::command]
async fn restart_node(state: tauri::State<'_, Shell>) -> Result<RunState, String> {
    let mut node = state.node.lock().await;
    node.stop().await.map_err(display)?;
    // The daemon clears its endpoint file as it goes; starting before that
    // lands would find the old file and attach to a process that is leaving.
    tokio::time::sleep(Duration::from_millis(400)).await;
    node.start().map_err(display)
}

#[tauri::command]
async fn set_limits(
    state: tauri::State<'_, Shell>,
    request: LimitsRequest,
) -> Result<LimitsUpdate, String> {
    let mut node = state.node.lock().await;
    node.set_limits(request).await.map_err(|error| {
        // `{:#}` keeps the chain: "those limits would not be valid: cpu_percent
        // must be 0-100" is actionable where either half alone is not.
        format!("{error:#}")
    })
}

fn display(error: anyhow::Error) -> String {
    format!("{error:#}")
}

/// Build the tray menu for a model and hang it on the tray icon.
fn apply_tray(app: &AppHandle, model: &TrayModel) -> tauri::Result<()> {
    let Some(icon) = app.tray_by_id(TRAY_ID) else {
        return Ok(());
    };
    let mut built = Vec::new();
    for item in &model.items {
        built.push(
            MenuItemBuilder::with_id(item.id.clone(), &item.label)
                .enabled(item.enabled)
                .build(app)?,
        );
    }
    let mut menu = MenuBuilder::new(app);
    for (index, item) in built.iter().enumerate() {
        menu = menu.item(item);
        // A rule under the status line, and another above Quit, so the
        // destructive item is not adjacent to the one people click most.
        if index == 0 || index + 2 == built.len() {
            menu = menu.separator();
        }
    }
    icon.set_menu(Some(menu.build()?))?;
    icon.set_tooltip(Some(&model.tooltip))?;
    icon.set_icon(Some(Image::from_bytes(icon_bytes(tray::icon_name(
        model.health,
    )))?))?;
    Ok(())
}

fn show_dashboard(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(WINDOW_ID) {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// Refresh the tray and any open window, forever.
async fn refresh_loop(app: AppHandle) {
    loop {
        let shell = app.state::<Shell>().inner().clone();
        let (snapshot, blocker) = {
            let mut node = shell.node.lock().await;
            let snapshot = node.snapshot(None).await;
            let coordinator = node.settings().coordinator.clone();
            let binary = node.supervisor_mut().node_binary().map(PathBuf::from);
            let blocker = start_blocker(binary.as_deref(), &coordinator);
            (snapshot, blocker)
        };
        let model = TrayModel::from_snapshot(&snapshot, blocker.as_deref());
        let _ = apply_tray(&app, &model);
        // The page listens for this and redraws. It also polls on its own, so
        // a missed event costs a few seconds of staleness rather than a frozen
        // dashboard.
        let _ = tauri::Emitter::emit(&app, "hocmesh://snapshot", &snapshot);
        tokio::time::sleep(REFRESH).await;
    }
}

/// Start the desktop app. Returns only when the operator quits.
pub fn run() -> anyhow::Result<()> {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            snapshot,
            get_settings,
            save_settings,
            start_node,
            stop_node,
            restart_node,
            set_limits
        ])
        .setup(|app| {
            let config_dir = app
                .path()
                .app_config_dir()
                .unwrap_or_else(|_| PathBuf::from("."));
            let shell: Shell = Arc::new(Shared::new(config_dir));
            app.manage(shell.clone());

            TrayIconBuilder::with_id(TRAY_ID)
                .icon(Image::from_bytes(ICON_STOPPED)?)
                .tooltip("hocMESH — starting")
                .on_menu_event(|app, event| {
                    let app = app.clone();
                    let id = event.id().0.clone();
                    tauri::async_runtime::spawn(async move {
                        handle_menu(&app, &id).await;
                    });
                })
                .on_tray_icon_event(|tray, event| {
                    // Left click opens the dashboard, which is what every
                    // tray app on every platform has trained people to expect.
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_dashboard(tray.app_handle());
                    }
                })
                .build(app)?;

            // Bring the node up with the app when asked, the way Docker
            // Desktop starts its engine with its window.
            let handle = app.handle().clone();
            let autostart = shell.clone();
            tauri::async_runtime::spawn(async move {
                let start = {
                    let node = autostart.node.lock().await;
                    node.settings().start_node_with_app
                };
                if start {
                    let mut node = autostart.node.lock().await;
                    if let Err(error) = node.start() {
                        tracing::warn!("could not start the node with the app: {error:#}");
                    }
                }
                refresh_loop(handle).await;
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the window leaves the app in the tray. The node keeps
            // running either way; this only decides whether the operator has
            // to relaunch to see it.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .build(tauri::generate_context!())?
        .run(|app, event| {
            if let RunEvent::ExitRequested { api, code, .. } = &event {
                // A tray app with no windows open is still running. Only an
                // explicit Quit -- which sets an exit code -- ends it.
                if code.is_none() {
                    api.prevent_exit();
                }
            }
            let _ = app;
        });
    Ok(())
}

async fn handle_menu(app: &AppHandle, id: &str) {
    let shell = app.state::<Shell>().inner().clone();
    match id {
        tray::ID_DASHBOARD => show_dashboard(app),
        tray::ID_START => {
            let mut node = shell.node.lock().await;
            if let Err(error) = node.start() {
                tracing::warn!("start from the tray failed: {error:#}");
            }
        }
        tray::ID_STOP => {
            let mut node = shell.node.lock().await;
            if let Err(error) = node.stop().await {
                tracing::warn!("stop from the tray failed: {error:#}");
            }
        }
        tray::ID_RESTART => {
            let mut node = shell.node.lock().await;
            if let Err(error) = node.stop().await {
                tracing::warn!("restart could not stop the node: {error:#}");
                return;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
            if let Err(error) = node.start() {
                tracing::warn!("restart could not start the node: {error:#}");
            }
        }
        tray::ID_QUIT => {
            // Only a node this app started goes down with it. One the operator
            // started elsewhere is left alone -- see `should_stop_on_quit`.
            let mut node = shell.node.lock().await;
            let home = node.settings().home.clone();
            if node.supervisor_mut().should_stop_on_quit(&home)
                && let Err(error) = node.stop().await
            {
                tracing::warn!("could not stop the supervised node on quit: {error:#}");
            }
            app.exit(0);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dashboard::Health;

    #[test]
    fn every_health_state_resolves_to_real_icon_bytes() {
        for health in [Health::Working, Health::Degraded, Health::Stopped] {
            let bytes = icon_bytes(tray::icon_name(health));
            assert!(!bytes.is_empty());
            assert_eq!(
                &bytes[..8],
                b"\x89PNG\r\n\x1a\n",
                "the tray icons must be real PNGs or the tray silently shows nothing"
            );
        }
    }

    #[test]
    fn the_three_tray_icons_are_actually_different_images() {
        // Compiled-in bytes make it easy to point two states at one file and
        // never notice: the tray would then be unable to say which state it is
        // in, which is the only thing it exists to do.
        assert_ne!(ICON_WORKING, ICON_DEGRADED);
        assert_ne!(ICON_WORKING, ICON_STOPPED);
        assert_ne!(ICON_DEGRADED, ICON_STOPPED);
    }

    #[test]
    fn an_unknown_icon_name_falls_back_to_stopped_rather_than_panicking() {
        assert_eq!(icon_bytes("nonsense.png"), ICON_STOPPED);
    }
}
