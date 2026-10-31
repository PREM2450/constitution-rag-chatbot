/* Constitution RAG — chat UI logic.
 *
 * Talks to the same-origin backend:
 *   GET  /health  (optional status poll)
 *   POST /ask     { question, retrieval_mode? } -> AnswerResponse
 *
 * The UI never invents citations: everything it displays comes from the
 * backend response fields (answer, citations, sources, warning, ...).
 */

(function () {
  "use strict";

  var messagesEl = document.getElementById("chat-messages");
  var formEl = document.getElementById("chat-form");
  var inputEl = document.getElementById("chat-input");
  var sendEl = document.getElementById("chat-send");
  var hintEl = document.getElementById("chat-hint");
  var statusEl = document.getElementById("backend-status");

  var pending = false;

  /* ---------- helpers ---------- */

  function el(tag, className, textContent) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (textContent !== undefined && textContent !== null) node.textContent = textContent;
    return node;
  }

  function scrollToEnd() {
    messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
  }

  function addUserMessage(text) {
    var wrap = el("div", "message message--user");
    wrap.appendChild(el("div", "message__bubble", text));
    messagesEl.appendChild(wrap);
    scrollToEnd();
  }

  function removeLoadingBubble() {
    var loading = messagesEl.querySelector(".message--loading");
    if (loading) loading.remove();
  }

  function addLoadingMessage() {
    var wrap = el("div", "message message--assistant message--loading");
    var bubble = el("div", "message__bubble");
    bubble.appendChild(el("span", null, "Searching the Constitution"));
    var dots = el("span", "dots");
    dots.appendChild(el("span"));
    dots.appendChild(el("span"));
    dots.appendChild(el("span"));
    bubble.appendChild(dots);
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToEnd();
  }

  function describeSource(source) {
    var meta = [];
    if (source.article) meta.push(source.article);
    if (source.page !== null && source.page !== undefined) {
      meta.push("PDF page " + source.page);
    }
    return meta.join(", ") || "Constitution of India";
  }

  function addAnswerMessage(payload) {
    var wrap = el("div", "message message--assistant");
    var bubble = el("div", "message__bubble");

    bubble.appendChild(el("div", null, payload.answer || "No answer returned."));

    if (payload.warning) {
      var warning = el("div", "answer-warning");
      warning.textContent = "⚠ " + payload.warning;
      bubble.appendChild(warning);
    }

    buildCitations(payload, bubble);
    buildSources(payload, bubble);

    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToEnd();
  }

  function buildCitations(payload, bubble) {
    var citations = (payload && payload.citations) || [];
    if (!citations.length) return;

    var box = el("div", "citations");

    citations.forEach(function (citation) {
      var chip = el("span", "citation-chip");
      var label = citation.article ? citation.article : citation.chunk_id;
      chip.appendChild(el("span", null, label));
      if (citation.page !== null && citation.page !== undefined) {
        chip.appendChild(el("span", "citation-page", " · p. " + citation.page));
      }
      chip.title = citation.source_title || "Constitution of India";
      box.appendChild(chip);
    });
    bubble.appendChild(box);
  }

  function buildSources(payload, bubble) {
    var sources = (payload && payload.sources) || [];
    if (!sources.length) return;

    var details = el("details", "sources-toggle");
    var count = sources.length;
    details.appendChild(el("summary", null, "Sources used (" + count + ")"));

    var chunks = {};
    sources.forEach(function (source) {
      if (source && source.chunk) chunks[source.chunk.chunk_id] = source.chunk;
    });

    Object.keys(chunks).forEach(function (chunkId) {
      var chunk = chunks[chunkId];
      var passage = el("div", "source-passage");
      passage.appendChild(
        el("div", "source-meta", describeSource(chunk) + (chunkId ? " · " + chunkId : ""))
      );
      passage.appendChild(el("div", null, chunk.text || ""));
      details.appendChild(passage);
    });

    bubble.appendChild(details);
  }

  function addErrorBubble(text) {
    var wrap = el("div", "message message--assistant");
    var bubble = el("div", "message__bubble");
    var error = el("div", "answer-error");
    error.textContent = "✕ " + text;
    bubble.appendChild(error);
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToEnd();
  }

  /* ---------- backend communication ---------- */

  function askQuestion(question) {
    addLoadingMessage();
    pending = true;
    setInputState(true);

    fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, retrieval_mode: "dense" }),
    })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return { detail: "The server returned an unreadable response." };
          })
          .then(function (data) {
            return { ok: response.ok, status: response.status, data: data };
          });
      })
      .then(function (result) {
        removeLoadingBubble();
        if (!result.ok) {
          addErrorBubble(translateError(result));
          return;
        }
        addAnswerMessage(result.data);
      })
      .catch(function () {
        removeLoadingBubble();
        addErrorBubble(
          "The answer service is currently unavailable. Please check that the backend is running and try again."
        );
      })
      .then(function () {
        pending = false;
        setInputState(false);
      });
  }

  function translateError(result) {
    var status = result.status;
    var detail = (result.data && result.data.detail) || null;
    if (detail) return detail;

    if (status === 503) {
      return "The retrieval service is not ready yet. The Constitution index is probably still being built. Please try again later.";
    }
    if (status === 422) return "The request was rejected by the server.";
    if (status === 405) return "This request method is not allowed.";
    return "The answer service returned HTTP " + status + ". Please try again later.";
  }

  /* ---------- input handling ---------- */

  function setInputState(disabled) {
    sendEl.disabled = disabled;
    inputEl.disabled = disabled;
    if (disabled) {
      hintEl.textContent = "Waiting for the answer…";
    } else {
      hintEl.textContent = "Press Enter to send, Shift + Enter for a new line.";
      inputEl.focus();
    }
  }

  function submitQuestion() {
    var question = inputEl.value.trim();
    if (!question) {
      addErrorBubble("Enter a question to get started.");
      return;
    }
    if (pending) return;

    /* keep the existing system hint only until the first exchange */
    var systemHint = messagesEl.querySelector(".message--system");
    if (systemHint) systemHint.remove();

    addUserMessage(question);
    inputEl.value = "";
    autoGrow();
    askQuestion(question);
  }

  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  }

  formEl.addEventListener("submit", function (event) {
    event.preventDefault();
    submitQuestion();
  });

  inputEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitQuestion();
    }
  });

  inputEl.addEventListener("input", autoGrow);

  /* ---------- backend status poll (best effort) ---------- */

  fetch("/health")
    .then(function (response) { return response.json(); })
    .then(function (health) {
      if (health.status === "ok") {
        statusEl.textContent = "Backend connected";
        statusEl.classList.add("app-footer__status--ok");
      } else {
        statusEl.textContent = "Backend running but retrieval is degraded: " +
          ((health.retrieval.retriever && health.retrieval.retriever.reason) || "unknown");
        statusEl.classList.add("app-footer__status--degraded");
      }
    })
    .catch(function () {
      statusEl.textContent = "Backend unreachable";
      statusEl.classList.add("app-footer__status--degraded");
    });

  sendEl.disabled = false;
})();
