import { useEffect, useMemo, useRef, useState } from "react";

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
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [typingMessageId, setTypingMessageId] = useState(null);
  const typingIntervalRef = useRef(null);
  const pendingTextRef = useRef("");
  const displayedTextRef = useRef("");

  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) {
        window.clearInterval(typingIntervalRef.current);
      }
    };
  }, []);

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

  const updateBotText = (messageId, text) => {
    setMessages((previous) =>
      previous.map((message) =>
        message.id === messageId
          ? {
              ...message,
              text,
            }
          : message,
      ),
    );
  };

  const waitForTypingToFinish = () =>
    new Promise((resolve) => {
      const check = () => {
        if (!pendingTextRef.current.length) {
          if (typingIntervalRef.current) {
            window.clearInterval(typingIntervalRef.current);
            typingIntervalRef.current = null;
          }
          resolve();
          return;
        }

        window.setTimeout(check, 20);
      };

      check();
    });

  const startTypewriter = (messageId) => {
    if (typingIntervalRef.current) {
      window.clearInterval(typingIntervalRef.current);
    }

    typingIntervalRef.current = window.setInterval(() => {
      if (!pendingTextRef.current.length) {
        return;
      }

      const nextChar = pendingTextRef.current.slice(0, 1);
      pendingTextRef.current = pendingTextRef.current.slice(1);
      displayedTextRef.current += nextChar;
      updateBotText(messageId, displayedTextRef.current);
    }, 14);
  };

  const sendMessage = async (event) => {
    event.preventDefault();

    const text = inputValue.trim();
    if (!text || isSending) {
      return;
    }

    setErrorMessage("");
    setIsSending(true);
    const botMessageId = `bot_${Date.now()}`;
    pendingTextRef.current = "";
    displayedTextRef.current = "";
    setTypingMessageId(botMessageId);
    setMessages((previous) => [
      ...previous,
      { id: `user_${Date.now()}`, role: "user", text },
      { id: botMessageId, role: "bot", text: "" },
    ]);
    setInputValue("");
    startTypewriter(botMessageId);

    try {
      let response;
      if (attachedFiles.length > 0) {
        const formData = new FormData();
        formData.append("message", text);
        formData.append("thread_id", threadId);
        attachedFiles.forEach((file) => {
          formData.append("files", file);
        });

        response = await fetch("/api/chat/extract/upload", {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch("/api/chat/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: text,
            thread_id: threadId,
          }),
        });
      }

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Failed to send message");
      }

      if (attachedFiles.length > 0) {
        const payload = await response.json().catch(() => ({}));
        const answer = payload.response || "No response from bot";
        pendingTextRef.current += answer;
        await waitForTypingToFinish();
        setTypingMessageId(null);
        setAttachedFiles([]);
        return;
      }

      if (!response.body) {
        throw new Error("Streaming is not supported in this browser");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let answer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        answer += chunk;
        pendingTextRef.current += chunk;
      }

      const finalChunk = decoder.decode();
      answer += finalChunk;
      pendingTextRef.current += finalChunk;
      await waitForTypingToFinish();
      setTypingMessageId(null);

      if (!answer.trim()) {
        updateBotText(botMessageId, "No response from bot");
      }
      setAttachedFiles([]);
    } catch (error) {
      if (typingIntervalRef.current) {
        window.clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
      pendingTextRef.current = "";
      displayedTextRef.current = "";
      setTypingMessageId(null);
      setMessages((previous) =>
        previous.filter((message) => message.id !== botMessageId),
      );
      setErrorMessage(error.message || "Could not reach chat API");
    } finally {
      if (typingIntervalRef.current) {
        window.clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
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
              className={`message message-${message.role}${
                message.role === "bot" && !message.text
                  ? " message-loading"
                  : ""
              }${message.id === typingMessageId ? " message-typing" : ""}`}
            >
              <span>{message.text || "Thinking..."}</span>
              {message.id === typingMessageId ? (
                <span className="message-cursor" aria-hidden="true">
                  |
                </span>
              ) : null}
            </article>
          ))}
        </div>

        <form className="chat-input-row" onSubmit={sendMessage}>
          <label className="file-input-label" htmlFor="document-upload">
            Attach document
          </label>
          <input
            id="document-upload"
            type="file"
            multiple
            onChange={(event) =>
              setAttachedFiles(Array.from(event.target.files || []))
            }
          />
          {attachedFiles.length > 0 ? (
            <p className="attachment-info">
              {attachedFiles.length} file(s) attached for extraction
            </p>
          ) : null}
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
