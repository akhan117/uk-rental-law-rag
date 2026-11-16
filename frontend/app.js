const conversation = document.getElementById("conversation");
const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const factsToggle = document.getElementById("facts-toggle");
const factsFields = document.getElementById("facts-fields");

document.getElementById("snapshot-date").textContent = new Date().toISOString().slice(0, 10);

factsToggle.addEventListener("click", () => {
  const expanded = factsToggle.getAttribute("aria-expanded") === "true";
  factsToggle.setAttribute("aria-expanded", String(!expanded));
  factsFields.hidden = expanded;
  factsToggle.textContent = expanded ? "+ Add known facts (optional)" : "- Hide facts";
});

function collectFacts() {
  const facts = {
    tenancy_type: document.getElementById("f-tenancy-type").value.trim() || null,
    location: document.getElementById("f-location").value.trim() || null,
    tenancy_start_date: document.getElementById("f-start-date").value.trim() || null,
    notice_details: document.getElementById("f-notice").value.trim() || null,
  };
  return Object.values(facts).some(Boolean) ? facts : null;
}

function renderSourceItem(source) {
  const li = document.createElement("li");
  const details = document.createElement("details");
  details.className = "source-item";
  const summary = document.createElement("summary");
  if (source.url) {
    const a = document.createElement("a");
    a.href = source.url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = source.citation || source.chunk_id;
    summary.appendChild(a);
  } else {
    summary.textContent = source.citation || source.chunk_id;
  }
  const text = document.createElement("div");
  text.className = "source-text";
  text.textContent = source.text;
  details.append(summary, text);
  li.appendChild(details);
  return li;
}

function renderAnswer(data) {
  const tpl = document.getElementById("answer-template");
  const node = tpl.content.cloneNode(true);
  node.querySelector(".short-answer-text").textContent = data.short_answer;
  node.querySelector(".applicable-law-text").textContent = data.applicable_law;
  node.querySelector(".how-applies-text").textContent = data.how_it_applies;
  if (data.uncertainties) {
    const block = node.querySelector(".uncertainties-block");
    block.hidden = false;
    node.querySelector(".uncertainties-text").textContent = data.uncertainties;
  }
  const list = node.querySelector(".sources-list");
  (data.sources || []).forEach((s) => list.appendChild(renderSourceItem(s)));
  return node;
}

function renderAbstain(data) {
  const tpl = document.getElementById("abstain-template");
  const node = tpl.content.cloneNode(true);
  node.querySelector(".abstain-text").textContent = data.short_answer;
  node.querySelector(".abstain-reason").textContent = data.scope_note || "";
  if (data.sources && data.sources.length) {
    const block = node.querySelector(".near-misses");
    block.hidden = false;
    const list = node.querySelector(".sources-list");
    data.sources.forEach((s) => list.appendChild(renderSourceItem(s)));
  }
  return node;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  const echo = document.createElement("div");
  echo.className = "question-echo";
  echo.textContent = question;
  conversation.appendChild(echo);

  const loading = document.createElement("p");
  loading.className = "loading";
  conversation.appendChild(loading);

  const submitBtn = form.querySelector("button");
  submitBtn.disabled = true;
  questionInput.value = "";

  // Answers take most of a minute on a machine without a GPU, nearly all of it
  // in the model. A static spinner is indistinguishable from a crash over that
  // long, so show the stage and a running clock: the wait is then evidently
  // progress rather than a hang.
  const started = Date.now();
  let stage = "Searching the legislation and case law…";
  const tick = () => {
    const secs = Math.round((Date.now() - started) / 1000);
    loading.textContent = secs < 3 ? stage : `${stage} (${secs}s)`;
  };
  tick();
  const timer = setInterval(tick, 1000);

  let preview = null;
  try {
    await streamQuery(question, collectFacts(), {
      onSources: (data) => {
        // Real evidence on screen while the model is still writing.
        stage = "Reading the sources and writing the answer…";
        tick();
        preview = renderSourcesPreview(data);
        conversation.insertBefore(preview, loading);
        conversation.scrollIntoView({ behavior: "smooth", block: "end" });
      },
      onAnswer: (data) => {
        if (preview) preview.remove();
        conversation.appendChild(data.abstained ? renderAbstain(data) : renderAnswer(data));
      },
      onError: (detail) => {
        loading.className = "loading error";
        loading.textContent = "Something went wrong talking to the backend: " + detail;
      },
    });
  } catch (err) {
    loading.className = "loading error";
    loading.textContent = "Something went wrong talking to the backend: " + err;
  } finally {
    clearInterval(timer);
    if (!loading.classList.contains("error")) loading.remove();
    submitBtn.disabled = false;
    conversation.scrollIntoView({ behavior: "smooth", block: "end" });
  }
});

/** Read the Server-Sent Events stream from /query/stream. */
async function streamQuery(question, facts, handlers) {
  const res = await fetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, facts }),
  });
  if (!res.ok || !res.body) throw new Error("HTTP " + res.status);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line; keep any partial frame buffered.
    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let event = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      const data = JSON.parse(dataLines.join("\n"));

      if (event === "sources") handlers.onSources(data);
      else if (event === "answer") handlers.onAnswer(data);
      else if (event === "error") handlers.onError(data.detail || "unknown error");
    }
  }
}

/** The retrieved sources, shown while the answer is still being generated. */
function renderSourcesPreview(data) {
  const wrap = document.createElement("div");
  wrap.className = "sources-preview";

  const heading = document.createElement("p");
  heading.className = "sources-preview-heading";
  const n = (data.sources || []).length;
  heading.textContent = `Found ${n} relevant source${n === 1 ? "" : "s"}:`;
  wrap.appendChild(heading);

  const list = document.createElement("ul");
  list.className = "sources-list";
  (data.sources || []).forEach((s) => list.appendChild(renderSourceItem(s)));
  wrap.appendChild(list);
  return wrap;
}

// Tell the user which hardware this is running on. Without a GPU an answer takes
// most of a minute instead of a few seconds, and Ollama falls back to CPU
// silently - so a slow session is otherwise indistinguishable from a broken one.
// The check is deliberately loud when there's no GPU and quiet when there is.
(async function reportRuntime() {
  const note = document.getElementById("first-run-note");
  const warning = document.getElementById("gpu-warning");
  if (!note || !warning) return;

  // The backend learns this from its startup warm-up, which takes a few seconds,
  // so the page can load before the answer is known. Poll briefly rather than
  // give up and leave the user with no indication either way.
  for (let attempt = 0; attempt < 15; attempt++) {
    let info;
    try {
      const res = await fetch("/health");
      if (!res.ok) return;
      info = await res.json();
    } catch {
      return; // The page works fine without this; never block on it.
    }

    if (info.ready && info.gpu !== null) {
      if (info.gpu) {
        // On its own line: this is the counterpart to the no-GPU warning, and
        // saying answers "should be fast" sets the expectation that a long wait
        // means something is wrong rather than being normal.
        note.appendChild(document.createElement("br"));
        const span = document.createElement("span");
        span.className = "runtime-good";
        span.textContent = "Running on your graphics card, so answers should be fast.";
        note.appendChild(span);
      } else {
        warning.hidden = false;
      }
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
})();
