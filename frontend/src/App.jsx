import { useState, useEffect, useRef } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [aiReady, setAiReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {}, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const addMessages = (newMessage, isUser) => {
    setMessages((prev) => [
      ...prev,
      { content: newMessage, isUser, id: Date.now() + Math.random() },
    ]);
  };

  const sendMessage = () => {
    if (inputValue.trim() === "") return;

    addMessages(inputValue, true);
    setInputValue("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-900 via-gray-900 to-neutral-800 flex flex-col items-center justify-center p-4 gap-8">
      <h1 className="text-6xl sm:text-7xl font-light text-gray-200 text-center">
        Chat
      </h1>
      <div className="w-full max-w-2xl bg-gradient-to-r from-gray-800/90 to-gray-700/90 backdrop-blur-md border border-gray-600 rounded-3xl p-6 shadow-2xl">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            👋 Aloita keskustelu kirjoittamalla viesti alla olevaan kenttään.
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`p-3 m-2 rounded-2xl max-w-xs whitespace-pre-wrap ${
              msg.isUser
                ? "bg-[#E1007A]/40 text-white self-end"
                : "bg-gray-600/40 text-gray-100 self-start"
            }`}
          >
            <div className="whitespace-pre-wrap">{msg.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef}></div>
      </div>
      <div className="flex flex-col items-center gap-4 w-full max-w-md mt-4">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Kirjoita viesti..."
          className="flex-1 px-4 py-3 bg-gray-700/80 border border-gray-600 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:shadow-xl focus:shadow-sky-400/80 focus:ring-sky-500 transition duration-400 disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          onClick={sendMessage}
          className="flex-1 px-4 py-3 bg-[#E1007A] hover:bg-[#c9006a] border border-gray-600 rounded-2xl text-white font-semibold shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-[#E1007A]/50 transition duration-200"
        >
          Lähetä viesti
        </button>
      </div>
    </div>
  );
}

export default App;
