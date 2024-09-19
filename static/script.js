document.getElementById("send-button").addEventListener("click", async () => {
    const input = document.getElementById("user-input");
    const message = input.value;

    // Add user message to chat history
    addMessage("User", message);

    // Clear input field
    input.value = "";

    // Send message to the backend
    const response = await fetch("/chat/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message })
    });

    const data = await response.json();
    addMessage("Bot", data.response);
});

function addMessage(sender, message) {
    const chatHistory = document.getElementById("chat-history");
    const messageElement = document.createElement("div");
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the bottom
}