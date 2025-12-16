let keyEvents = [];
let startTime = 0;

// Track key events
document.addEventListener("keydown", (e) => {
  if (keyEvents.length === 0) startTime = performance.now();
  keyEvents.push({
    key: e.key,
    t: performance.now() - startTime,
    type: "down"
  });
});

document.addEventListener("keyup", (e) => {
  keyEvents.push({
    key: e.key,
    t: performance.now() - startTime,
    type: "up"
  });
});

// Handle form submission
function submitData() {
  const userId = document.getElementById("user_id").value || "unknown";
  const sentence = document.getElementById("sentence").value.trim();

  if (!sentence) {
    alert("Please type the sentence before submitting!");
    return;
  }

  const payload = {
    user_id: userId,
    events: keyEvents,
    timestamp: Date.now()
  };

  // Send to both: save_log (for training data) and predict (for live result)
  Promise.all([
    fetch("/api/save_log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
  ])
    .then(async ([saveRes, predictRes]) => {
      const predictData = await predictRes.json();
      if (predictData.predicted_user) {
        document.getElementById("result").innerText =
          ` Predicted User: ${predictData.predicted_user}`;
      } else {
        document.getElementById("result").innerText =
          ` ${predictData.error || "Prediction failed."}`;
      }

      // reset
      keyEvents = [];
      document.getElementById("sentence").value = "";
    })
    .catch((err) => {
      console.error("Error:", err);
      document.getElementById("result").innerText =
       " Something went wrong.";
    });
}
