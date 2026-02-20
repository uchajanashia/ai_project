const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendButton");

let loadingRow = null;

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendMessage(role, text) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;

  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollToBottom();
  return row;
}

function showLoading() {
  loadingRow = document.createElement("div");
  loadingRow.className = "message-row assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble assistant loading";
  bubble.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';

  loadingRow.appendChild(bubble);
  chatMessages.appendChild(loadingRow);
  scrollToBottom();
}

function hideLoading() {
  if (loadingRow) {
    loadingRow.remove();
    loadingRow = null;
  }
}

function setLoadingState(isLoading) {
  chatInput.disabled = isLoading;
  sendButton.disabled = isLoading;
}

function autosizeInput() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 140)}px`;
}

chatInput.addEventListener("input", autosizeInput);

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = chatInput.value.trim();
  if (!question) return;

  appendMessage("user", question);
  chatInput.value = "";
  autosizeInput();

  setLoadingState(true);
  showLoading();

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question })
    });

    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = {};
    }

    if (!response.ok) {
      const detail =
        typeof payload.detail === "string"
          ? payload.detail
          : "Request failed.";
      throw new Error(detail);
    }

    const answer =
      typeof payload.answer === "string" && payload.answer.trim()
        ? payload.answer.trim()
        : "No answer returned.";

    hideLoading();
    appendMessage("assistant", answer);
  } catch (error) {
    hideLoading();
    appendMessage("assistant", `Error: ${error.message}`);
  } finally {
    setLoadingState(false);
    chatInput.focus();
  }
});

appendMessage("assistant", "Ready. Ask a question to query the RAG backend.");
chatInput.focus();
