import { useMemo, useState } from "react";

function App() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "bot",
      text: "Hi! Ask me anything.",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const threadId = useMemo(() => {
    // Keep a stable thread id during a browser session.
    const key = "chat_thread_id";
    const existing = window.sessionStorage.getItem(key);
    if (existing) {
      return existing;
    }

    const next = `thread_${crypto.randomUUID()}`;
    window.sessionStorage.setItem(key, next);
    return next;
  }, []);

  const sendMessage = async (event) => {
    event.preventDefault();

    const text = inputValue.trim();
    if (!text || isSending) {
      return;
    }

    setErrorMessage("");
    setIsSending(true);
    setMessages((previous) => [
      ...previous,
      { id: `user_${Date.now()}`, role: "user", text },
    ]);
    setInputValue("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: text,
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Failed to send message");
      }

      const payload = await response.json();
      setMessages((previous) => [
        ...previous,
        {
          id: `bot_${Date.now()}`,
          role: "bot",
          text: payload.answer || "No response from bot",
        },
      ]);
    } catch (error) {
      setErrorMessage(error.message || "Could not reach chat API");
    } finally {
      setIsSending(false);
    }
  };

  return (
    <main className="chat-page">
      <section className="chat-card">
        <header className="chat-header">
          <h1>LangGraph Chat</h1>
          <p>Talk to your backend agent in real time.</p>
        </header>

        <div className="chat-messages" aria-live="polite">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message message-${message.role}`}
            >
              <span>{message.text}</span>
            </article>
          ))}
          {isSending ? (
            <article className="message message-bot message-loading">
              <span>Thinking...</span>
            </article>
          ) : null}
        </div>

        <form className="chat-input-row" onSubmit={sendMessage}>
          <input
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder="Type your message"
            aria-label="Message"
          />
          <button type="submit" disabled={isSending || !inputValue.trim()}>
            Send
          </button>
        </form>

        {errorMessage ? <p className="chat-error">{errorMessage}</p> : null}
      </section>
    </main>
  );
}

export default App;
