function toggleChat() {
  const chatbox = document.getElementById("chatbot");
  chatbox.style.display = chatbox.style.display === "flex" ? "none" : "flex";
}
function sendMessage() {
  const input = document.getElementById("userInput");
  const text = input.value.trim();
  if (!text) return;
  const chatBody = document.getElementById("chat-body");
  const userMsg = document.createElement("div");
  userMsg.className = "user-message";
  userMsg.innerText = text;
  chatBody.appendChild(userMsg);
  input.value = "";
  setTimeout(() => {
    const botMsg = document.createElement("div");
    botMsg.className = "bot-message";
    botMsg.innerText = getBotResponse(text);
    chatBody.appendChild(botMsg);
    chatBody.scrollTop = chatBody.scrollHeight;
  }, 1000);
}
function getBotResponse(msg) {
  const lower = msg.toLowerCase();
  if (lower.includes("salary")) return "Your predicted salary depends on experience and current CTC.";
  if (lower.includes("hello")) return "Hello! Ask me about employee salary prediction!";
  if (lower.includes("predict")) return "Use the input sliders to predict your salary.";
  return "I’m still learning 🤖. Ask about salary, AI, experience!";
}
