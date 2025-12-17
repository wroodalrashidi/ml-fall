/* ======================================================
   ELEMENTS
====================================================== */
const tableBody = document.getElementById("eventsBody");
const alertSound = document.getElementById("fallAlert");
const enableBtn = document.getElementById("enableAlerts");

/* ======================================================
   STATE
====================================================== */
// store seen FALL event IDs
const seenFallEvents = new Set();
let audioEnabled = false;

/* ======================================================
   AUDIO UNLOCK (REQUIRED BY BROWSER)
====================================================== */
enableBtn.addEventListener("click", async () => {
  try {
    // silent unlock
    alertSound.volume = 0;
    await alertSound.play();
    alertSound.pause();
    alertSound.currentTime = 0;
    alertSound.volume = 1;

    audioEnabled = true;
    enableBtn.style.display = "none";
    console.log("🔓 Audio unlocked");
  } catch (e) {
    console.error("❌ Audio unlock failed", e);
  }
});

/* ======================================================
   TIME FORMAT (KUWAIT)
====================================================== */
function formatKuwaitTime(ts) {
  if (!ts) return "—";

  // already human-readable (CSV case)
  if (typeof ts === "string" && ts.includes("/")) return ts;

  const date = new Date(ts);
  if (isNaN(date.getTime())) return "—";

  return date.toLocaleString("en-GB", {
    timeZone: "Asia/Kuwait",
    hour12: false,
  });
}

/* ======================================================
   LOAD EVENTS
====================================================== */
async function loadEvents() {
  const res = await fetch("/events", {cache: "no-store"});
  const events = await res.json();

  tableBody.innerHTML = "";
  if (!events.length) return;

  let newFallDetected = false;

  /* ---------- TABLE ---------- */
  events.forEach((e) => {
    const isFall = e.fall === "True";

    // 🔔 detect NEW fall row
    if (isFall && !seenFallEvents.has(e.id)) {
      seenFallEvents.add(e.id);
      newFallDetected = true;
    }

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${e.id || "—"}</td>
      <td>${e.image || "—"}</td>
      <td>
        <span class="${isFall ? "badge-fall-true" : "badge-fall-false"}">
          ${isFall ? "FALL" : "NORMAL"}
        </span>
      </td>
      <td>${formatKuwaitTime(e.timestamp)}</td>
    `;

    tableBody.appendChild(row);
  });

  /* ---------- ALERT SOUND (4s ONLY) ---------- */
  if (newFallDetected && audioEnabled) {
    alertSound.currentTime = 0;
    alertSound.play().catch(() => {});

    // stop after 4 seconds
    setTimeout(() => {
      alertSound.pause();
      alertSound.currentTime = 0;
    }, 4000);
  }

  /* ---------- SNAPSHOT ---------- */
  const fallEvents = events.filter((e) => e.fall === "True");

  document.getElementById("totalImages").textContent = events.length;
  document.getElementById("fallCount").textContent = fallEvents.length;

  document.getElementById("lastFallDetected").textContent =
    fallEvents.length > 0 ? formatKuwaitTime(fallEvents[0].timestamp) : "—";
}

/* ======================================================
   INIT
====================================================== */
loadEvents();
setInterval(loadEvents, 5000);
