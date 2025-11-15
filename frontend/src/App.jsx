import { useState, useEffect, useRef } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const [error, setError] = useState("");
  const normalizeText = (text) => text.replace(/[\s\-_.]/g, "").toUpperCase();

  useEffect(() => {}, []);


  //Vierittää keskustelunäkymän automaattisesti viimeiseen viestiin
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };


  // useEffect varmistaa, että aina kun viestilista (messages) muuttuu,
  // näkymä vieritetään automaattisesti alas uuteen viestiin.
  useEffect(scrollToBottom, [messages]);


  // Lisää uuden viestin tilaan (messages)
  // Luo viestille uniikin id:n, liittää aikaleiman ja yhdistää sen olemassa olevaan listaan.
  const addMessages = (message) => {
    const now = new Date();
    const timestamp = `${now.getHours()}.${String(now.getMinutes()).padStart(2, "0")}`;
    setMessages((prev) => [
      ...prev,
      { id: Date.now() + Math.random(), ...message, timestamp, },
    ]);
  };

  // Tarkistaa sisältääkö käyttäjän syöte arkaluonteisia tietoja
  const containsSensitiveData = (text) => {
    const normalized = normalizeText(text);

    // Henkilötunnuksen tiukka ja löysempi tunnistus
    const socPattern = /\d{6}[+\-A]\d{3}[0-9A-Y]/;
    const socLoosePattern = /\d{6}\d{3}[0-9A-Y]/;

    // Sähköpostin ja puhelinnumeron tunnistus
    const emailPattern = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i;
    const phonePattern = /(?:\+358|0)\d{9,}/;

    // Palauttaa true jos jokin ehto täyttyy
    return (
      socPattern.test(normalized) ||
      socLoosePattern.test(normalized) ||
      phonePattern.test(normalized) ||
      emailPattern.test(text)
    );
  };

  // Lähettää käyttäjän viestin palvelimelle ja käsittelee vastauksen
  const sendMessage = async () =>{
    const userText = inputValue.trim();
    if(userText === "") return;

    if(containsSensitiveData(inputValue)){
      setError("⚠️ Älä kirjoita henkilötietoja, sähköpostiosoitetta tai puhelinnumeroa.");
      return;
    }

    // Nollataan mahdollinen virheviesti
    setError("");

    // Lisätään käyttäjän viesti viestilistaan
    addMessages({ content: userText, isUser: true });
    setInputValue("");
    setIsLoading(true);

    try {
      // Lähetetään viesti backendille POST-pyynnönä
      
    const response = await fetch("http://localhost:8000/api/llm/query",{
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ question: userText }),
      
    });

    // Jos vastaus ei ole OK, heitetään virhe
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    const data = await response.json();
    //console.log("✅ Testivastaus:", data);
    
    // Lisätään tekoälyn vastaus viestilistaan
    addMessages({ content: data.answer, isUser: false });
  } catch (error) {
    console.error("Virhe LLM-kyselyssä:", error);
    // Jos pyyntö epäonnistuu, lisätään virheilmoitus keskusteluun
    addMessages({content: "⚠️ En saanut vastausta palvelimelta.", isUser: false});
  } finally {
    setIsLoading(false);
  }

  };

  // Käsittelee näppäimistön painallukset syötekentässä
  const handleKeyPress = (e) => {
    // Tarkistetaan, painettiinko Enter-näppäintä ilman Shift-näppäintä
    if (e.key === "Enter" && !e.shiftKey) {
      // Estetään oletustoiminto (rivinvaihto tekstikentässä)
      e.preventDefault();
      // Lähetetään viesti kutsumalla sendMessage-funktiota
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-900 via-gray-900 to-neutral-800 flex flex-col items-center justify-center p-4 gap-8">
      <h1 className="text-6xl sm:text-7xl font-light text-gray-200 text-center">
        Chat
      </h1>
      <div className={`flex items-center gap 2 px-4 py-1.5 rounded-full text-sm font-semibold shadow-md border transition duration-200 ${isLoading ? "bg-[#E1007A]/20 border border-[#E1007]/30 text-pink-200"  :
        "bg-[#E1007A]/20 border border-[#E1007]/30 text-pink-200"
      }`}>
        {isLoading ? "⏳ Odotetaan vastausta..." : "🟢 AI valmis"}
      </div>
      <div className="w-full max-w-2xl bg-gradient-to-r from-gray-800/90 to-gray-700/90 backdrop-blur-md border border-gray-600 rounded-3xl p-6 shadow-2xl">
        <div className="h-[32rem] overflow-y-auto border-b boprder-gray-600 mb-6 p-4 bg-gradient-to-b from-gray-900/50 to-gray-800-50 rounded-2xl">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-20">
              👋 Aloita keskustelu kirjoittamalla viesti alla olevaan kenttään.
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.isUser ? "items-start" : "items-end"}`}
            >
              <div className={`p-3 m-2 rounded-2xl break-words whitespace-pre-wrap overflow-hidden ${
                    msg.isUser
                      ? "w-[60%] bg-[#E1007A]/40 text-white"
                      : "w-[80%] bg-gray-600/40 text-gray-100"
                  }`}
                >
                  {msg.content}
              </div>
              <div className={`text-xs text-gray-400 mb-2 ${msg.isUser ? "ml-4 text-left" : "mr-4 text-right"}`}>
                {msg.timestamp}
              </div>
          </div>
          ))}
          
          <div ref={messagesEndRef}></div>
        </div>
        <div className="flex flex-col items-center gap-4 w-full mx-auto mt-4">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              if (error && !containsSensitiveData(e.target.value)) setError("");
            }}
            onKeyDown={handleKeyPress}
            placeholder="Kirjoita viesti..."
            className="w-full px-4 py-3 bg-gray-700/80 border border-gray-600 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:shadow-xl focus:shadow-sky-400/80 focus:ring-sky-500 transition duration-400 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {error && <p className="text-red-400 text-sm">{error}</p>}
          <button
            onClick={sendMessage}
            disabled={isLoading}
            className="w-full px-4 py-3 bg-[#E1007A] hover:bg-[#c9006a] border border-gray-600 rounded-2xl text-white font-semibold shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-[#E1007A]/50 transition duration-200"
          >
            Lähetä viesti
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;