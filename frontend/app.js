// DOM Elements
const recordButton = document.getElementById('recordButton');
const resetButton = document.getElementById('resetButton');
const statusText = document.getElementById('statusText');
const recordingDot = document.getElementById('recordingDot');
const chatHistory = document.getElementById('chatHistory');
const errorMsg = document.getElementById('errorMsg');

// State variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// We use base64 audio from backend, store them here to play on demand
const audioCache = {};
let audioIdCounter = 0;

const API_URL = 'http://localhost:8000/chat/';

// Request Microphone Access
async function setupAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = []; // Reset
            await sendAudioToBackend(audioBlob);
        };

        statusText.textContent = "SYSTEM READY / WAITING ...";

    } catch (err) {
        console.error("Microphone error:", err);
        showError("MICROPHONE ACCESS DENIED.");
        recordButton.disabled = true;
    }
}

// Button Click Handler
recordButton.addEventListener('click', () => {
    if (!mediaRecorder) return;

    if (!isRecording) {
        // Start conversation
        audioChunks = [];
        mediaRecorder.start();
        isRecording = true;

        recordButton.textContent = "STOP & TRANSLATE";
        recordButton.classList.add('recording');
        statusText.textContent = "LISTENING CONVERSATION ...";
        recordingDot.classList.remove('hidden');
        errorMsg.classList.add('hidden');

        appendSystemMessage("Recording started...");
        scrollToBottom();

    } else {
        // Stop and process
        mediaRecorder.stop();
        isRecording = false;

        recordButton.textContent = "START CONVERSATION";
        recordButton.classList.remove('recording');
        recordButton.disabled = true;

        statusText.textContent = "ANALYZING VOICES & TRANSLATING ...";
        recordingDot.classList.add('hidden');
    }
});

// Reset Speakers Button
resetButton.addEventListener('click', async () => {
    try {
        const response = await fetch('http://localhost:8000/reset_speakers/', {
            method: 'POST'
        });
        if (response.ok) {
            // Clear chat history
            chatHistory.innerHTML = '';
            appendSystemMessage("SPEAKER MEMORY CLEARED. READY FOR NEW CONVERSATION.");
            statusText.textContent = "SPEAKERS RESET.";
        }
    } catch (error) {
        showError("RESET FAILED: " + error.message);
    }
});

// Play base64 audio
function playAudio(id) {
    if (audioCache[id]) {
        const audio = new Audio("data:audio/mp3;base64," + audioCache[id]);
        audio.play();
    }
}

// Attach to window object so inline HTML onclick can find it
window.playAudio = playAudio;

// Send Blob to Diarization Backend
async function sendAudioToBackend(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'chat.webm');

    try {
        appendSystemMessage("Uploading to AI core...");
        scrollToBottom();

        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `SERVER ERROR (${response.status})`);
        }

        const data = await response.json();

        if (!data.dialogues || data.dialogues.length === 0) {
            appendSystemMessage("No translatable speech detected.");
            return;
        }

        // Process each dialogue segment
        data.dialogues.forEach((dlg) => {
            appendChatMessage(dlg);
        });

        scrollToBottom();
        statusText.textContent = "TRANSLATION COMPLETE.";

    } catch (error) {
        console.error("API Error:", error);
        showError("ERROR: " + error.message);
        statusText.textContent = "FAILED.";
    } finally {
        recordButton.disabled = false;
    }
}

function appendSystemMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-msg';
    msgDiv.innerHTML = `<div class="msg-content">${text}</div>`;
    chatHistory.appendChild(msgDiv);
}

function appendChatMessage(dialogue) {
    const msgDiv = document.createElement('div');

    // Determine CSS class based on speaker ID.
    // Extract number from SPEAKER_XX format for dynamic styling
    let speakerNum = 0;
    const match = dialogue.speaker.match(/(\d+)$/);
    if (match) {
        speakerNum = parseInt(match[1], 10);
    }
    let cssSpeakerClass = `speaker-${speakerNum % 2}`;
    let displayName = `SPEAKER ${speakerNum + 1}`;

    msgDiv.className = `message ${cssSpeakerClass}`;

    // Store Audio in memory map and get an ID
    const audioId = `audio_${audioIdCounter++}`;
    audioCache[audioId] = dialogue.audio_b64;

    msgDiv.innerHTML = `
        <div class="msg-header">${displayName}</div>
        <div class="msg-content">
            ${dialogue.translated}
            <div class="msg-original">${dialogue.original}</div>
            <button class="btn-play-audio" onclick="playAudio('${audioId}')">▶ PLAY AUDIO</button>
        </div>
    `;

    chatHistory.appendChild(msgDiv);

    // Auto-play the first translated segment (or all sequentially if you implement a queue)
    // For simplicity, we just auto-play if it's the very first bubble rendered in this batch.
    // playAudio(audioId);
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove('hidden');
}

// Initialize on page load
window.addEventListener('load', setupAudio);
