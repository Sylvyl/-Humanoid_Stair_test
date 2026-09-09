let csrf = null;
let heartbeatTimer = null;

const $ = (id) => document.getElementById(id);
const fmtM = (value) => value == null ? "—" : `${(value * 1000).toFixed(0)} mm`;
const api = async (path, method = "GET", body = null) => {
  const options = { method, headers: {} };
  if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  if (csrf) options.headers["X-CSRF-Token"] = csrf;
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `${response.status}`);
  return value;
};

function render(state) {
  const robot = state.telemetry.robot;
  const connection = $("connection");
  connection.textContent = robot.connected ? "PC2 CONNECTED" : "PC2 OFFLINE";
  connection.className = `pill ${robot.connected ? "good" : "bad"}`;
  $("mission-state").textContent = state.mission.state;
  $("mission-copy").textContent = state.mission.fault || (state.commands_compiled
    ? "Control adapter available; all gates still apply."
    : "Shadow mode is enforced. This build cannot publish robot commands.");
  $("confidence").textContent = state.stairs.available ? `${Math.round(state.stairs.confidence * 100)}%` : "—";
  $("direction").textContent = state.stairs.direction?.toUpperCase() || "—";
  $("riser").textContent = fmtM(state.stairs.riser_m);
  $("tread").textContent = fmtM(state.stairs.tread_m);
  $("width").textContent = fmtM(state.stairs.width_m);
  $("power-source").textContent = state.compatibility.roles.power
    ? state.compatibility.roles.power.split("/").pop().toUpperCase()
    : "MISSING";
  const stamps = Object.values(state.telemetry.sensors).map((x) => x.stamp || 0);
  const newest = stamps.length ? Math.max(...stamps) : 0;
  $("sensor-age").textContent = newest ? `${Math.max(0, Date.now()/1000-newest).toFixed(1)} s` : "—";
  const blockers = state.safety.blockers;
  $("blockers").innerHTML = blockers.length
    ? blockers.map((x) => `<div class="blocker">${escapeHtml(x)}</div>`).join("")
    : '<div class="all-clear">All software gates currently pass.</div>';
  const gate = state.release_gate;
  $("gate-state").textContent = gate.unlocked ? "UNLOCKED" : "LOCKED";
  $("gate-state").className = `pill ${gate.unlocked ? "good" : "bad"}`;
  $("gate-blockers").innerHTML = gate.blockers.length
    ? gate.blockers.map((x) => `<div class="blocker">${escapeHtml(x)}</div>`).join("")
    : '<div class="all-clear">Evidence permits the next supervised hardware stage.</div>';
  $("raw-state").textContent = JSON.stringify(state, null, 2);
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

async function refresh() {
  try { render(await api("/api/v1/state")); }
  catch (error) { $("notice").textContent = `Refresh failed: ${error.message}`; }
}

async function post(path, body = {}) {
  try {
    const state = await api(path, "POST", body);
    render(state);
    $("notice").textContent = "Action accepted in the safety supervisor.";
  } catch (error) {
    $("notice").textContent = error.message;
  }
}

async function loadLessons() {
  const lessons = await api("/api/v1/lessons");
  $("lessons").innerHTML = lessons.map((item, index) => `<article class="lesson"><div class="number">${String(index+1).padStart(2,"0")}</div><div><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.expected)}</p></div></article>`).join("");
  $("lesson-count").textContent = `0 / ${lessons.length}`;
}

$("login").addEventListener("click", async () => {
  try {
    const value = await api("/api/v1/session/login", "POST", { token: $("token").value });
    csrf = value.csrf;
    $("login-box").hidden = true;
    $("controls").hidden = false;
    $("token").value = "";
    $("notice").textContent = "Authenticated for 15 minutes.";
  } catch (error) { $("notice").textContent = error.message; }
});
$("save-checks").addEventListener("click", () => post("/api/v1/checklist", {
  hardware_estop: $("check-estop").checked,
  gantry: $("check-gantry").checked,
  spotter: $("check-spotter").checked,
}));
$("arm").addEventListener("click", () => post("/api/v1/control/arm"));
$("start-up").addEventListener("click", () => post("/api/v1/control/start", { mission_id: `web-${Date.now()}`, direction: "up" }));
$("start-down").addEventListener("click", () => post("/api/v1/control/start", { mission_id: `web-${Date.now()}`, direction: "down" }));
$("disarm").addEventListener("click", () => post("/api/v1/control/disarm"));
$("software-stop").addEventListener("click", () => post("/api/v1/control/abort"));

const deadman = $("deadman");
const beginHeartbeat = async () => {
  if (heartbeatTimer) return;
  deadman.classList.add("active");
  await post("/api/v1/control/heartbeat");
  heartbeatTimer = setInterval(() => post("/api/v1/control/heartbeat"), 100);
};
const endHeartbeat = () => {
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  heartbeatTimer = null;
  deadman.classList.remove("active");
};
deadman.addEventListener("pointerdown", beginHeartbeat);
for (const event of ["pointerup", "pointercancel", "pointerleave"]) deadman.addEventListener(event, endHeartbeat);
window.addEventListener("blur", endHeartbeat);

loadLessons().catch((error) => $("notice").textContent = error.message);
refresh();
setInterval(refresh, 1000);
