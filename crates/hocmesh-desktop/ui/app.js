/* hocMESH Desktop -- the window.
 *
 * This file draws a snapshot and sends back button presses. It computes
 * nothing: every number and every label arrives already formatted from Rust,
 * where the rounding rules are covered by tests. If you find yourself about to
 * do arithmetic here, do it in `dashboard.rs` instead.
 */

const invoke = window.__TAURI__.core.invoke;
const listen = window.__TAURI__.event.listen;

const REFRESH_MS = 3000;

/** The most recent snapshot, or null before the first one lands. */
let snap = null;
/** Older ledger pages the operator asked for, newest page first. */
let olderPages = [];
/** True while the operator is mid-edit, so a refresh does not yank the sliders. */
let limitsDirty = false;
let busy = false;

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  // textContent, never innerHTML: some of these strings come from a
  // coordinator this machine does not control.
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function card(label, value, sub, tone) {
  const box = el("div", "card");
  box.append(el("div", "card-label", label));
  box.append(el("div", "card-value" + (tone ? " is-" + tone : ""), value));
  if (sub) box.append(el("div", "card-sub", sub));
  return box;
}

function kv(list, pairs) {
  list.replaceChildren();
  for (const [key, value] of pairs) {
    if (value === null || value === undefined || value === "") continue;
    list.append(el("dt", null, key));
    list.append(el("dd", null, value));
  }
}

let toastTimer = null;
function toast(message, bad) {
  const box = $("toast");
  box.textContent = message;
  box.classList.toggle("is-bad", Boolean(bad));
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    box.hidden = true;
  }, bad ? 7000 : 3000);
}

function flash(id) {
  const mark = $(id);
  mark.hidden = false;
  setTimeout(() => {
    mark.hidden = true;
  }, 1800);
}

/* ---------------- navigation ---------------- */

for (const button of document.querySelectorAll(".nav-item")) {
  button.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".nav-item")) {
      other.classList.toggle("is-active", other === button);
    }
    for (const view of document.querySelectorAll(".view")) {
      view.classList.toggle("is-active", view.id === "view-" + button.dataset.view);
    }
  });
}

/* ---------------- rendering ---------------- */

function renderTopbar(s) {
  const o = s.overview;
  $("state-dot").className = "dot is-" + o.health;
  $("state-label").textContent = o.health_label;
  $("state-sub").textContent = o.running
    ? `${o.coordinator} · up ${o.uptime || "moments"}${o.supervised ? " · started by this app" : ""}`
    : o.coordinator;

  $("btn-start").disabled = busy || o.running;
  $("btn-stop").disabled = busy || !o.running;
  $("btn-restart").disabled = busy || !o.running;

  $("foot-app-version").textContent = o.app_version;
  $("foot-node-version").textContent = o.node_version || "—";

  const banner = $("banner-error");
  if (o.last_error) {
    banner.textContent = o.last_error;
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }
}

function renderOverview(s) {
  const o = s.overview;
  const stats = $("overview-stats");
  stats.replaceChildren(
    card("Earned this run", o.earned_this_run + " CU", "since the node started"),
    card("Jobs completed", o.jobs_completed, o.workers ? `${o.workers} workers` : null, "good"),
    card("Jobs failed", o.jobs_failed, null, o.jobs_failed > 0 ? "bad" : null),
    card("Inferences", o.inferences_completed, o.ai_offered ? "AI offered" : "AI not offered"),
    card("Uptime", o.uptime || "—"),
    card("Last contact", o.last_contact, "with the coordinator"),
  );

  kv($("overview-identity"), [
    ["Node ID", o.node_id || "not created yet — start the node once"],
    ["Coordinator", o.coordinator],
    ["Node version", o.node_version || "not running"],
    ["Desktop version", o.app_version],
    ["Process", o.running ? (o.supervised ? "started by this app" : "started elsewhere") : "stopped"],
  ]);
}

function renderResources(s) {
  const r = s.resources;
  $("banner-restart").hidden = !r.restart_required;

  if (!limitsDirty) {
    $("lim-cpu").value = r.cpu_percent;
    $("lim-mem").value = r.memory_percent;
    $("lim-gpu").value = r.gpu_percent;
    $("lim-ai").checked = r.ai === true;
  }
  paintLimitNotes(s);

  $("lim-ai-note").textContent = r.ai_effective
    ? "Offered to the mesh now."
    : r.ai === true
      ? "Consented, but not offered: no runtime is installed, or no graphics share is lent."
      : "Not offered. Your own local use of AI is unaffected.";

  kv($("machine-facts"), [
    ["Host", r.hostname],
    ["Processor", r.cpu_brand],
    ["System", `${r.os} · ${r.arch}`],
    ["Logical processors", `${r.shared_logical_cpus} of ${r.logical_cpus} lent`],
    ["Memory", `${r.shared_memory} of ${r.total_memory} lent`],
  ]);

  const list = $("accelerators");
  list.replaceChildren();
  if (!r.accelerators.length) {
    list.append(el("p", "hint", "No accelerators detected on this machine."));
  } else {
    const wrap = el("div", "table-wrap");
    const table = el("table", "table");
    const head = el("thead");
    const headRow = el("tr");
    for (const label of ["Device", "Vendor", "Backend", "Memory"]) {
      headRow.append(el("th", null, label));
    }
    head.append(headRow);
    const body = el("tbody");
    for (const gpu of r.accelerators) {
      const row = el("tr");
      row.append(el("td", null, gpu.is_cpu ? gpu.name + " (fallback)" : gpu.name));
      row.append(el("td", null, gpu.vendor));
      row.append(el("td", null, gpu.backend));
      row.append(el("td", null, gpu.memory || "—"));
      body.append(row);
    }
    table.append(head, body);
    wrap.append(table);
    list.append(wrap);
  }
}

function paintLimitNotes(s) {
  const r = s.resources;
  const cpu = Number($("lim-cpu").value);
  const mem = Number($("lim-mem").value);
  const gpu = Number($("lim-gpu").value);
  $("lim-cpu-value").textContent = cpu + "%";
  $("lim-mem-value").textContent = mem + "%";
  $("lim-gpu-value").textContent = gpu + "%";
  $("lim-cpu-note").textContent = `${r.shared_logical_cpus} of ${r.logical_cpus} logical processors`;
  $("lim-mem-note").textContent = `${r.shared_memory} of ${r.total_memory}`;
  $("lim-gpu-note").textContent = r.accelerators.length
    ? `${r.accelerators.length} device${r.accelerators.length === 1 ? "" : "s"} detected`
    : "no accelerator detected";
}

function ledgerRows(s) {
  const rows = [];
  if (s.ledger) rows.push(...s.ledger.entries);
  for (const page of olderPages) rows.push(...page);
  const seen = new Set();
  return rows.filter((row) => {
    const key = row.sequence != null ? "s" + row.sequence : [row.when, row.delta, row.reason, row.reference].join("|");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function renderLedger(s) {
  const totals = $("ledger-totals");
  if (s.ledger) {
    totals.replaceChildren(
      card("Balance", s.ledger.balance + " CU", "available to spend"),
      card("Earned", s.ledger.earned + " CU", "work done for others", "good"),
      card("Spent", s.ledger.spent + " CU", "work others did for you"),
      card("Ledger height", s.ledger.ledger_height ?? "—", s.ledger.authoritative ? "validator chain" : "coordinator record"),
    );
    $("ledger-source").textContent = s.ledger.authoritative
      ? "Signed by validators"
      : "Coordinator's own record";
  } else {
    totals.replaceChildren();
    $("ledger-source").textContent = "";
  }

  const body = $("ledger-body");
  body.replaceChildren();
  const rows = s.ledger ? ledgerRows(s) : [];

  if (!rows.length) {
    const row = el("tr");
    const cell = el("td", "empty", s.ledger_error || "Nothing on this node's ledger yet.");
    cell.colSpan = 4;
    row.append(cell);
    body.append(row);
  } else {
    for (const entry of rows) {
      const row = el("tr");
      row.append(el("td", null, entry.when));
      row.append(el("td", null, entry.reason));
      row.append(el("td", null, entry.reference || "—"));
      row.append(el("td", "num " + (entry.positive ? "delta-up" : "delta-down"), entry.delta));
      body.append(row);
    }
  }

  const cursor = nextCursor(s);
  $("btn-ledger-more").disabled = busy || cursor === null;
  $("ledger-note").textContent = s.ledger_error
    ? s.ledger_error
    : rows.length
      ? `${rows.length} entr${rows.length === 1 ? "y" : "ies"} shown`
      : "";
}

function nextCursor(s) {
  if (olderPages.length) {
    const last = olderPages[olderPages.length - 1];
    return last.nextBefore ?? null;
  }
  return s.ledger ? (s.ledger.next_before ?? null) : null;
}

function render(s) {
  snap = s;
  renderTopbar(s);
  renderOverview(s);
  renderResources(s);
  renderLedger(s);
}

/* ---------------- actions ---------------- */

async function refresh() {
  try {
    render(await invoke("snapshot", { before: null }));
  } catch (error) {
    toast(String(error), true);
  }
}

async function withBusy(work, done) {
  if (busy) return;
  busy = true;
  if (snap) renderTopbar(snap);
  try {
    await work();
    if (done) toast(done);
  } catch (error) {
    toast(String(error), true);
  } finally {
    busy = false;
    await refresh();
  }
}

$("btn-start").addEventListener("click", () =>
  withBusy(() => invoke("start_node"), "Node started."),
);
$("btn-stop").addEventListener("click", () => withBusy(() => invoke("stop_node"), "Node stopped."));
$("btn-restart").addEventListener("click", () =>
  withBusy(() => invoke("restart_node"), "Node restarted."),
);

for (const id of ["lim-cpu", "lim-mem", "lim-gpu"]) {
  $(id).addEventListener("input", () => {
    limitsDirty = true;
    if (snap) paintLimitNotes(snap);
  });
}
$("lim-ai").addEventListener("change", () => {
  limitsDirty = true;
});

$("btn-reset-limits").addEventListener("click", () => {
  limitsDirty = false;
  if (snap) renderResources(snap);
});

$("btn-apply-limits").addEventListener("click", () =>
  withBusy(async () => {
    await invoke("set_limits", {
      request: {
        cpu_percent: Number($("lim-cpu").value),
        memory_percent: Number($("lim-mem").value),
        gpu_percent: Number($("lim-gpu").value),
        ai: $("lim-ai").checked,
      },
    });
    limitsDirty = false;
    flash("limits-saved");
  }),
);

$("btn-ledger-more").addEventListener("click", () =>
  withBusy(async () => {
    const cursor = snap ? nextCursor(snap) : null;
    if (cursor === null) return;
    const page = await invoke("snapshot", { before: cursor });
    if (!page.ledger) return;
    const entries = page.ledger.entries;
    entries.nextBefore = page.ledger.next_before ?? null;
    olderPages.push(entries);
  }),
);

/* ---------------- settings ---------------- */

async function loadSettings() {
  try {
    const settings = await invoke("get_settings");
    $("set-home").value = settings.home;
    $("set-coordinator").value = settings.coordinator;
    $("set-workers").value = settings.workers ?? "";
    $("set-port").value = settings.control_port;
    $("set-autostart").checked = settings.start_node_with_app;
    $("set-noai").checked = settings.no_ai;
  } catch (error) {
    toast(String(error), true);
  }
}

$("btn-save-settings").addEventListener("click", () =>
  withBusy(async () => {
    const workers = $("set-workers").value.trim();
    await invoke("save_settings", {
      settings: {
        home: $("set-home").value.trim(),
        coordinator: $("set-coordinator").value.trim(),
        start_node_with_app: $("set-autostart").checked,
        workers: workers === "" ? null : Number(workers),
        no_ai: $("set-noai").checked,
        control_port: Number($("set-port").value || 0),
      },
    });
    await loadSettings();
    flash("settings-saved");
  }),
);

/* ---------------- start ---------------- */

listen("hocmesh://snapshot", (event) => {
  if (!busy) render(event.payload);
});

loadSettings();
refresh();
// The event stream is the fast path; this is the seatbelt for a missed one.
setInterval(() => {
  if (!busy) refresh();
}, REFRESH_MS);
