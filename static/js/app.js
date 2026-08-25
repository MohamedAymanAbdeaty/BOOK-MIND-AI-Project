const state = { books: [], activeBook: null, busy: false, chatHistories: {} };

const elements = {
  library: document.querySelector("#libraryView"),
  chat: document.querySelector("#chatView"),
  grid: document.querySelector("#bookGrid"),
  messages: document.querySelector("#messages"),
  form: document.querySelector("#chatForm"),
  question: document.querySelector("#question"),
  send: document.querySelector("#sendButton"),
  activeCover: document.querySelector("#activeCover"),
  activeTitle: document.querySelector("#activeTitle"),
  activeAuthor: document.querySelector("#activeAuthor"),
};

const suggestionMap = {
  rich_dad_poor_dad: ["How does the book define an asset?", "What is financial intelligence?"],
  the_art_of_war: ["Why is preparation important?", "What does it mean to win without fighting?"],
  meditations: ["What can we control?", "What does the book say about good character?"],
};

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value ?? "";
  return node.innerHTML;
}
function coverMarkup(book) {
  if (book.cover_image) {
    return `<img src="${escapeHtml(book.cover_image)}" alt="${escapeHtml(book.title)} cover" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;">`;
  }
  return `<span class="cover-top">BookMind edition</span><span class="cover-monogram">${escapeHtml(book.monogram)}</span><span class="cover-title">${escapeHtml(book.title)}</span>`;
}

function renderBooks() {
  elements.grid.innerHTML = state.books.map((book, index) => `
    <button class="book-card" type="button" data-book="${book.id}" aria-label="Ask ${escapeHtml(book.title)}">
      <span class="book-cover cover-${book.accent}">${coverMarkup(book)}</span>
      <span class="book-info">
        <span class="book-number">0${index + 1}</span>
        <h3>${escapeHtml(book.title)}</h3>
        <p>${escapeHtml(book.description)}</p>
        <span class="ask-link">Open conversation ↗</span>
      </span>
    </button>
  `).join("");
}

function selectBook(bookId) {
  if (state.activeBook && state.activeBook.id !== bookId) {
    state.chatHistories[state.activeBook.id] = elements.messages.innerHTML;
  }

  state.activeBook = state.books.find((book) => book.id === bookId);
  if (!state.activeBook) return;

  if (state.chatHistories[bookId]) {
    elements.messages.innerHTML = state.chatHistories[bookId];
    elements.messages.scrollTop = elements.messages.scrollHeight;
  } else {
    resetConversation();
  }

  elements.activeCover.className = `mini-cover cover-${state.activeBook.accent}`;
  elements.activeCover.innerHTML = coverMarkup(state.activeBook);
  elements.activeTitle.textContent = state.activeBook.title;
  elements.activeAuthor.textContent = state.activeBook.author;
  
  const currentSuggestions = document.querySelector("#suggestions");
  if (currentSuggestions) {
    currentSuggestions.innerHTML = (suggestionMap[bookId] || []).map((text) => `<button type="button">${escapeHtml(text)}</button>`).join("");
  }
  
  elements.library.hidden = true;
  elements.chat.hidden = false;
  window.scrollTo({ top: 0 });
  elements.question.focus();
}

function showLibrary() {
  if (state.activeBook) {
    state.chatHistories[state.activeBook.id] = elements.messages.innerHTML;
  }
  elements.chat.hidden = true;
  elements.library.hidden = false;
  state.activeBook = null;
  window.scrollTo({ top: 0 });
}

function resetConversation() {
  if (state.activeBook) {
    delete state.chatHistories[state.activeBook.id];
  }
  const welcome = document.querySelector(".welcome-message");
  elements.messages.innerHTML = "";
  if (welcome) {
    elements.messages.append(welcome);
    welcome.hidden = false;
  }
}

function addMessage(role, content, payload = null) {
  document.querySelector(".welcome-message")?.setAttribute("hidden", "");
  const article = document.querySelector("#messageTemplate").content.firstElementChild.cloneNode(true);
  article.classList.add(role);
  article.querySelector(".message-meta").textContent = role === "user" ? "You" : "BookMind · Evidence response";
  const body = article.querySelector(".message-body");
  body.innerHTML = `<p>${escapeHtml(content)}</p>`;

  if (payload) {
    const approved = payload.review_verdict === "approved";
    const badge = document.createElement("div");
    badge.className = `review-badge ${approved ? "" : "refused"}`;
    badge.textContent = approved ? `✓ Reviewer verified${payload.cached ? " · cache hit" : ""}` : `◇ ${payload.review_verdict}`;
    body.append(badge);

    if (payload.pipeline?.length) {
      const pipeline = document.createElement("p");
      pipeline.className = "pipeline";
      pipeline.textContent = payload.pipeline.join("  →  ");
      body.append(pipeline);
    }

    if (payload.sources?.length) {
      const sources = document.createElement("div");
      sources.className = "source-list";
      sources.innerHTML = payload.sources.map((source, index) => `
        <details>
          <summary>Source ${index + 1} · ${source.page ? `Page ${source.page}` : "Location unavailable"}<span class="source-score">${Math.round(source.score * 100)}% match</span></summary>
          <blockquote>${escapeHtml(source.text)}</blockquote>
        </details>
      `).join("");
      body.append(sources);
    }
    if (!approved) article.classList.add("blocked");
  }

  elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return article;
}

function addThinking() {
  const article = addMessage("assistant", "");
  article.dataset.thinking = "true";
  article.querySelector(".message-body").innerHTML = `<div class="thinking"><span class="thinking-dots"><span></span><span></span><span></span></span><span>Searching the book and reviewing evidence…</span></div>`;
}

async function ask(question) {
  if (!state.activeBook || state.busy) return;
  state.busy = true;
  elements.send.disabled = true;
  addMessage("user", question);
  addThinking();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ book_id: state.activeBook.id, question }),
    });
    const payload = await response.json();
    document.querySelector('[data-thinking="true"]')?.remove();
    if (!response.ok) throw new Error(payload.message || "The service is unavailable right now.");
    addMessage("assistant", payload.answer, payload);
  } catch (error) {
    document.querySelector('[data-thinking="true"]')?.remove();
    addMessage("assistant", error.message, { review_verdict: "refused", pipeline: ["Request could not be completed"] });
  } finally {
    state.busy = false;
    elements.send.disabled = false;
    elements.question.focus();
  }
}

elements.grid.addEventListener("click", (event) => {
  const card = event.target.closest("[data-book]");
  if (card) selectBook(card.dataset.book);
});

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = elements.question.value.trim();
  if (!question) return;
  elements.question.value = "";
  elements.question.style.height = "auto";
  ask(question);
});

elements.question.addEventListener("input", () => {
  elements.question.style.height = "auto";
  elements.question.style.height = `${Math.min(elements.question.scrollHeight, 130)}px`;
});

elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

document.addEventListener("click", (event) => {
  const btn = event.target.closest("#suggestions button");
  if (btn) ask(btn.textContent);
});

document.querySelector("#backButton").addEventListener("click", showLibrary);
document.querySelector("#brandButton").addEventListener("click", showLibrary);
document.querySelector("#clearButton").addEventListener("click", resetConversation);

fetch("/api/books")
  .then((response) => response.json())
  .then((payload) => { state.books = payload.books; renderBooks(); })
  .catch(() => { elements.grid.innerHTML = "<p>The library could not be loaded.</p>"; });
