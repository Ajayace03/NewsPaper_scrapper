// Update current time
function updateTime() {
  const now = new Date();
  document.getElementById("current-time").textContent = now
    .toTimeString()
    .substring(0, 8);
}
setInterval(updateTime, 1000);
updateTime();

// Military quotes rotation
const tacticalQuotes = [
  '"In the field, information is ammunition." - Gen. Charles Krulak',
  '"Information dominance is the key to modern warfare." - Joint Chiefs Memo',
  '"The first casualty when war comes is truth." - Sen. Hiram Johnson',
  '"Know thy enemy and know yourself." - Sun Tzu',
  '"Intelligence is the first line of defense." - CIA Doctrine',
  '"INTEL FLASH: The newspaper is mightier than the sword."',
  '"Good intelligence is nine-tenths of any battle." - Napoleon Bonaparte',
  '"EYES ONLY: Knowledge precedes victory."',
  '"Time spent in reconnaissance is seldom wasted." - British Army Field Manual',
  '"Intel without action is just trivia." - Special Forces Handbook',
];

function rotateQuotes() {
  const quoteElement = document.getElementById("tactical-quote");
  const randomIndex = Math.floor(Math.random() * tacticalQuotes.length);
  quoteElement.textContent = tacticalQuotes[randomIndex];
}
setInterval(rotateQuotes, 10000);
rotateQuotes();

// DOM elements
const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-input");
const selectFileBtn = document.getElementById("select-file-btn");
const previewContainer = document.getElementById("preview-container");
const previewContainer1 = document.getElementById("preview-container1");

const imagePreview = document.getElementById("image-preview");
const imagePreview1 = document.getElementById("image-preview1");
const resultheadline = document.getElementById("resultheadline");
const processBtn = document.getElementById("process-btn");
const clearBtn = document.getElementById("clear-btn");
const extractionResult = document.getElementById("extraction-result");
const progressContainer = document.getElementById("progress-container");
const progressBar = document.getElementById("progress-bar");
const loader = document.getElementById("loader");
const statusMessage = document.getElementById("status-message");
const statusText = document.getElementById("status-text");
const playBtn = document.getElementById("play-btn");
const pauseBtn = document.getElementById("pause-btn");
const stopBtn = document.getElementById("stop-btn");
const voiceSelect = document.getElementById("voice-select");
const rateInput = document.getElementById("rate");
const rateValue = document.getElementById("rate-value");

// Update status
function updateStatus(message, isError = false) {
  statusText.textContent = message;
  if (isError) {
    statusText.style.color = "var(--alert-red)";
  } else {
    statusText.style.color = "var(--military-green)";
  }
}

// Drag and Drop handlers
["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
  dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

["dragenter", "dragover"].forEach((eventName) => {
  dropArea.addEventListener(eventName, highlight, false);
});

["dragleave", "drop"].forEach((eventName) => {
  dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
  dropArea.classList.add("active");
}

function unhighlight() {
  dropArea.classList.remove("active");
}

// Handle file drop
dropArea.addEventListener("drop", handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  handleFiles(files);
}

// File selection button
selectFileBtn.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", function () {
  handleFiles(this.files);
});

function handleFiles(files) {
  if (files.length > 0) {
    const file = files[0];
    if (file.type.startsWith("image/")) {
      displayPreview(file);
      updateStatus("DOCUMENT RECEIVED");
    } else {
      updateStatus("ERROR: NOT AN IMAGE FILE", true);
      alert(
        "MISSION ABORT: File is not an image. Please select an image file."
      );
    }
  }
}

// Display image preview
function displayPreview(file) {
  previewContainer.style.display = "block";
  const reader = new FileReader();

  reader.onload = function (e) {
    imagePreview1.src = e.target.result;
  };

  reader.readAsDataURL(file);
}

// Improved OCR text processing
function enhanceTextOutput(text) {
  if (!text || text.trim().length === 0) {
    return "ERROR: No text detected in image. Try a clearer image or adjust contrast before uploading.";
  }

  // Basic cleanup
  let cleaned = text
    .trim()
    .replace(/\s+/g, " ")
    .replace(/(\r\n|\n|\r)/gm, " ")
    .replace(/[^\w\s.,;:!?'"()-]/g, "")
    .replace(/\.([A-Z])/g, ". $1");

  // Fix common OCR errors
  cleaned = cleaned
    .replace(/l1/g, "h")
    .replace(/0/g, "o")
    .replace(/1/g, "i")
    .replace(/\b(\w{1,2})\s+(?=[a-z])/g, "$1")
    .replace(/\b(\w)[\s.]+(\w\w)\b/g, "$1$2");

  // Create paragraphs
  cleaned = cleaned
    .replace(/\.\s+(?=[A-Z])/g, ".\n\n")
    .replace(/([.!?])\s+/g, "$1 ")
    .replace(/\n{3,}/g, "\n\n");

  return militarizeText(cleaned);
}

// Add military styling to text
function militarizeText(text) {
  const paragraphs = text.split("\n\n").filter((p) => p.trim().length > 0);
  if (paragraphs.length === 0) return text;

  const militaryPrefixes = [
    "INTEL BRIEFING: ",
    "SITUATION REPORT: ",
    "FIELD INTELLIGENCE: ",
    "COMMAND UPDATE: ",
    "EYES ONLY: ",
    "CLASSIFIED INTEL: ",
    "RECONNAISSANCE DATA: ",
    "TACTICAL ANALYSIS: ",
    "MISSION CRITICAL: ",
    "PRIORITY TRANSMISSION: ",
  ];

  const militarySuffixes = [
    " OVER.",
    " COPY THAT?",
    " CONFIRM RECEIPT.",
    " MAINTAIN RADIO SILENCE.",
    " EXECUTE WITH EXTREME PREJUDICE.",
    " BE ADVISED.",
    " HOO-AH!",
    " STAY FROSTY.",
    " REMAIN VIGILANT.",
    " END TRANSMISSION.",
  ];

  // Add prefix to first paragraph
  paragraphs[0] =
    militaryPrefixes[Math.floor(Math.random() * militaryPrefixes.length)] +
    paragraphs[0];

  // Add random military phrases to some paragraphs
  for (let i = 1; i < paragraphs.length; i++) {
    if (Math.random() < 0.4) {
      const prefix =
        militaryPrefixes[Math.floor(Math.random() * militaryPrefixes.length)];
      paragraphs[i] = prefix + paragraphs[i];
    }

    if (Math.random() < 0.3) {
      const suffix =
        militarySuffixes[Math.floor(Math.random() * militarySuffixes.length)];
      paragraphs[i] = paragraphs[i] + suffix;
    }
  }

  // Military sign-off at the end
  paragraphs[paragraphs.length - 1] += " END OF INTELLIGENCE REPORT.";

  return paragraphs.join("\n\n");
}

// Perform OCR on image
function performOCR(imageSource) {
  progressContainer.style.display = "block";
  loader.style.display = "block";
  extractionResult.textContent = "INITIATING EXTRACTION PROTOCOL... STANDBY...";
  statusMessage.textContent = "INTELLIGENCE ACQUISITION IN PROGRESS";
  statusMessage.className = "status-msg status-success";
  updateStatus("PROCESSING DOCUMENT");

  Tesseract.recognize(imageSource, "eng", {
    logger: (progress) => {
      if (progress.status === "recognizing text") {
        const percentage = Math.round(progress.progress * 100);
        progressBar.style.width = `${percentage}%`;
      }
    },
  })
    .then(({ data: { text } }) => {
      // Process and display results
      const enhancedText = enhanceTextOutput(text);
      extractionResult.textContent = enhancedText;

      // Enable voice controls
      playBtn.disabled = false;
      pauseBtn.disabled = true;
      stopBtn.disabled = true;

      progressContainer.style.display = "none";
      loader.style.display = "none";
      statusMessage.textContent = "INTELLIGENCE ACQUISITION COMPLETE";
      statusMessage.className = "status-msg status-success";
      updateStatus("EXTRACTION COMPLETE");
    })
    .catch((error) => {
      extractionResult.textContent = `ERROR: EXTRACTION FAILURE - ${error.message}`;
      progressContainer.style.display = "none";
      loader.style.display = "none";
      statusMessage.textContent = "MISSION ABORTED: EXTRACTION FAILED";
      statusMessage.className = "status-msg status-error";
      updateStatus("EXTRACTION FAILED", true);
    });
}

// Process button handler
processBtn.addEventListener("click", function () {
  if (imagePreview1.src) {
    performOCR(imagePreview.src);
  } else {
    alert("No document selected. Please upload an image first.");
  }
});

// Clear button handler
clearBtn.addEventListener("click", function () {
  previewContainer.style.display = "none";
  imagePreview1.src = "";
  fileInput.value = "";
  extractionResult.textContent =
    "NO INTELLIGENCE RECEIVED. AWAITING DATA UPLOAD.";
  progressContainer.style.display = "none";
  progressBar.style.width = "0%";
  playBtn.disabled = true;
  pauseBtn.disabled = true;
  stopBtn.disabled = true;
  statusMessage.textContent = "";
  statusMessage.className = "status-msg";
  updateStatus("AWAITING MISSION");
});

// Speech synthesis setup
const synth = window.speechSynthesis;
let utterance = null;

// Populate voices dropdown
function populateVoiceList() {
  voiceSelect.innerHTML = "";
  const voices = synth.getVoices();

  // Filter for English voices if available
  const englishVoices = voices.filter((voice) => voice.lang.includes("en"));
  const voicesToUse = englishVoices.length > 0 ? englishVoices : voices;

  voicesToUse.forEach((voice) => {
    const option = document.createElement("option");
    option.textContent = `${voice.name} (${voice.lang})`;
    option.setAttribute("data-voice", voice.name);
    voiceSelect.appendChild(option);
  });

  // Select deeper voice by default if available
  const defaultVoice = voicesToUse.find(
    (voice) =>
      voice.name.toLowerCase().includes("male") ||
      voice.name.toLowerCase().includes("guy")
  );

  if (defaultVoice) {
    const options = Array.from(voiceSelect.options);
    const optionToSelect = options.find(
      (option) => option.getAttribute("data-voice") === defaultVoice.name
    );
    if (optionToSelect) {
      voiceSelect.selectedIndex = options.indexOf(optionToSelect);
    }
  }
}

// Initialize voice list
if (synth.onvoiceschanged !== undefined) {
  synth.onvoiceschanged = populateVoiceList;
}
populateVoiceList();
// Fallback for browsers that don't support onvoiceschanged
setTimeout(populateVoiceList, 1000);

// Handle speech rate change
rateInput.addEventListener("input", () => {
  rateValue.textContent = parseFloat(rateInput.value).toFixed(1);
});

// Text-to-speech function
function speakText() {
  if (synth.speaking) {
    synth.cancel();
  }

  const selectedVoice =
    voiceSelect.selectedOptions[0].getAttribute("data-voice");
  const voices = synth.getVoices();
  const voice = voices.find((v) => v.name === selectedVoice);

  utterance = new SpeechSynthesisUtterance(extractionResult.textContent);
  utterance.voice = voice;
  utterance.rate = parseFloat(rateInput.value);
  utterance.pitch = 0.8; // Deeper voice for military effect

  // Event handlers
  utterance.onstart = () => {
    playBtn.disabled = true;
    pauseBtn.disabled = false;
    stopBtn.disabled = false;
    statusMessage.textContent = "BROADCAST IN PROGRESS";
    statusMessage.className = "status-msg status-success";
    updateStatus("BROADCASTING");
  };

  utterance.onend = () => {
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    statusMessage.textContent = "BROADCAST COMPLETE";
    statusMessage.className = "status-msg status-success";
    updateStatus("BROADCAST COMPLETE");
  };

  utterance.onerror = (event) => {
    statusMessage.textContent = `BROADCAST ERROR: ${event.error}`;
    statusMessage.className = "status-msg status-error";
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    updateStatus("BROADCAST ERROR", true);
  };

  synth.speak(utterance);
}

// Voice control button handlers
playBtn.addEventListener("click", speakText);

pauseBtn.addEventListener("click", function () {
  if (synth.speaking) {
    if (synth.paused) {
      synth.resume();
      pauseBtn.textContent = "PAUSE TRANSMISSION";
      statusMessage.textContent = "BROADCAST RESUMED";
    } else {
      synth.pause();
      pauseBtn.textContent = "RESUME TRANSMISSION";
      statusMessage.textContent = "BROADCAST PAUSED";
    }
  }
});

stopBtn.addEventListener("click", function () {
  if (synth.speaking) {
    synth.cancel();
    playBtn.disabled = false;
    stopBtn.addEventListener("click", function () {
      if (synth.speaking) {
        synth.cancel();
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        stopBtn.disabled = true;
        pauseBtn.textContent = "PAUSE TRANSMISSION";
        statusMessage.textContent = "BROADCAST TERMINATED";
        statusMessage.className = "status-msg status-error";
        updateStatus("BROADCAST ABORTED");
      }
    });
  }
});

// Add keyboard shortcuts for control
document.addEventListener("keydown", function (event) {
  // Space bar to toggle play/pause
  if (
    event.code === "Space" &&
    document.activeElement.tagName !== "BUTTON" &&
    document.activeElement.tagName !== "INPUT"
  ) {
    event.preventDefault();
    if (synth.speaking) {
      if (synth.paused) {
        synth.resume();
        pauseBtn.textContent = "PAUSE TRANSMISSION";
      } else {
        synth.pause();
        pauseBtn.textContent = "RESUME TRANSMISSION";
      }
    } else if (!playBtn.disabled) {
      speakText();
    }
  }

  // Escape key to stop
  if (event.code === "Escape" && synth.speaking) {
    synth.cancel();
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    pauseBtn.textContent = "PAUSE TRANSMISSION";
  }
});

// Add easter egg - Konami code to toggle "ultra classified" mode
let konamiIndex = 0;
const konamiCode = [
  "ArrowUp",
  "ArrowUp",
  "ArrowDown",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "ArrowLeft",
  "ArrowRight",
  "b",
  "a",
];
document.addEventListener("keydown", function (event) {
  if (event.key === konamiCode[konamiIndex]) {
    konamiIndex++;
    if (konamiIndex === konamiCode.length) {
      document.body.style.color = "#ff3333";
      document
        .querySelectorAll(
          ".btn, .upload-area, .mission-briefing, .intelligence-panel, .content-display, .control-panel"
        )
        .forEach((el) => {
          el.style.borderColor = "#ff3333";
        });

      statusMessage.textContent = "ULTRAVIOLET CLEARANCE ACTIVATED";
      statusMessage.className = "status-msg status-error";

      // Reset index
      konamiIndex = 0;

      // Return to normal after 5 seconds
      setTimeout(() => {
        document.body.style.color = "var(--military-green)";
        document
          .querySelectorAll(
            ".btn, .upload-area, .mission-briefing, .intelligence-panel, .content-display, .control-panel"
          )
          .forEach((el) => {
            el.style.borderColor = "var(--military-green)";
          });
        statusMessage.textContent = "RETURNING TO STANDARD CLEARANCE";
        statusMessage.className = "status-msg status-success";
      }, 5000);
    }
  } else {
    konamiIndex = 0;
  }
});

// Add responsive window behavior
window.addEventListener("resize", function () {
  if (window.innerWidth < 768) {
    document.getElementById("tactical-quote").textContent =
      '"MOBILE OPS ACTIVE"';
  } else {
    rotateQuotes();
  }
});

// Initialize with system status
window.addEventListener("load", function () {
  updateStatus("SYSTEM INITIALIZED");
  setTimeout(() => {
    updateStatus("AWAITING MISSION");
  }, 2000);

  // Check for browser compatibility
  if (!window.FileReader) {
    statusMessage.textContent = "ERROR: BROWSER NOT COMPATIBLE";
    statusMessage.className = "status-msg status-error";
    updateStatus("SYSTEM ERROR", true);
  }

  if (!window.speechSynthesis) {
    statusMessage.textContent = "ERROR: BROADCAST SYSTEM OFFLINE";
    statusMessage.className = "status-msg status-error";
    document.getElementById("play-btn").disabled = true;
  }
});

// Add military jargon to terminal messages periodically
setInterval(() => {
  if (
    extractionResult.textContent ===
    "NO INTELLIGENCE RECEIVED. AWAITING DATA UPLOAD."
  ) {
    const jargon = [
      "STANDING BY FOR INTEL DROP...",
      "AWAITING STRATEGIC DATA...",
      "TRANSMISSION CHANNEL OPEN...",
      "SECURE UPLINK ESTABLISHED...",
      "READY FOR CLASSIFIED INPUT...",
    ];
    extractionResult.textContent =
      jargon[Math.floor(Math.random() * jargon.length)];

    // Revert back after 3 seconds
    setTimeout(() => {
      if (!imagePreview1.src) {
        extractionResult.textContent =
          "NO INTELLIGENCE RECEIVED. AWAITING DATA UPLOAD.";
      }
    }, 3000);
  }
}, 15000);

// Visual effect for typing animation in extraction result
function typeEffect(element, text, speed = 30) {
  let i = 0;
  element.textContent = "";
  const timer = setInterval(() => {
    if (i < text.length) {
      element.textContent += text.charAt(i);
      i++;
    } else {
      clearInterval(timer);
    }
  }, speed);
}

// Add typing effect to status updates
function typeStatus(message, isError = false) {
  updateStatus("", isError);
  let i = 0;
  const timer = setInterval(() => {
    if (i < message.length) {
      statusText.textContent += message.charAt(i);
      i++;
    } else {
      clearInterval(timer);
    }
  }, 50);
}

// Elements for displaying fetched content
document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const statusText = document.getElementById("status-text");
  const fetchButton = document.getElementById("current-status");
  const imagePreview = document.getElementById("image-preview");
  const extractionResult = document.getElementById("extraction-result");
  const playBtn = document.getElementById("play-btn");
  const pauseBtn = document.getElementById("pause-btn");
  const stopBtn = document.getElementById("stop-btn");
  const voiceSelect = document.getElementById("voice-select");
  const rateInput = document.getElementById("rate");
  const rateValue = document.getElementById("rate-value");
  const currentTimeDisplay = document.getElementById("current-time");

  // Update current time every second
  function updateTime() {
    const now = new Date();
    const timeString = now.toTimeString().split(" ")[0];
    currentTimeDisplay.textContent = timeString;
  }
  setInterval(updateTime, 1000);
  updateTime();

  // Initialize speech synthesis
  let speechSynth = window.speechSynthesis;
  let currentUtterance = null;

  // Populate voice selector
  function populateVoiceList() {
    const voices = speechSynth.getVoices();
    voiceSelect.innerHTML = "";

    voices.forEach((voice) => {
      const option = document.createElement("option");
      option.textContent = `${voice.name} (${voice.lang})`;
      option.setAttribute("data-lang", voice.lang);
      option.setAttribute("data-name", voice.name);
      voiceSelect.appendChild(option);
    });
  }

  populateVoiceList();
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = populateVoiceList;
  }

  // Update rate display
  rateInput.addEventListener("input", () => {
    rateValue.textContent = parseFloat(rateInput.value).toFixed(1);
  });

  // Function to fetch and display data
  async function fetchFiles() {
    try {
      // Display loading status
      statusText.textContent = "FETCHING DATA";
      fetchButton.disabled = true;

      // Fetch Image
      const imageResponse = await fetch("http://localhost:5000/files/image");
      if (!imageResponse.ok) throw new Error("Image fetch failed");
      const imageBlob = await imageResponse.blob();
      const imageURL = URL.createObjectURL(imageBlob);
      imagePreview.src = imageURL;
      document.getElementById("preview-container").style.display = "block";

      // Fetch JSON data
      const jsonResponse = await fetch("http://127.0.0.1:5000/files/json");
      if (!jsonResponse.ok) {
        throw new Error("Failed to fetch JSON data.");
      }
      const jsonData = await jsonResponse.json();
      console.log(jsonData);

      // Create structured HTML for the extraction result
      let resultHTML = document.getElementById("");

      // Process each item in the JSON
      Object.keys(jsonData).forEach((key) => {
        const item = jsonData[key];
        if (item.headlines) {
          resultHTML += `<div><h3 class="intel-headline">${item.headlines}</h3>`;
        }

        if (item.subheadlines) {
          resultHTML += `<h4 class="intel-subheadline">${item.subheadlines}</h4>`;
        }

        if (item.content) {
          resultHTML += `<p class="intel-content">${item.content}</p>`;
        }

        resultHTML += '<hr class="intel-divider">';
      });

      resultHTML += "</div>";

      // Display the formatted JSON Data
      extractionResult.innerHTML = resultHTML;

      // Enable audio controls
      playBtn.disabled = false;
      pauseBtn.disabled = true;
      stopBtn.disabled = true;

      // Success message
      statusText.textContent = "MISSION SUCCESSFUL";
      document
        .getElementById("current-status")
        .classList.remove("status-error");
      document.getElementById("current-status").classList.add("status-success");
    } catch (error) {
      console.error("Error fetching files:", error);

      extractionResult.innerHTML = `<p class="error-message">ERROR: Failed to retrieve intelligence data.<br>Details: ${error.message}</p>`;
      statusText.textContent = "MISSION FAILED";
      document
        .getElementById("current-status")
        .classList.remove("status-success");
      document.getElementById("current-status").classList.add("status-error");
    } finally {
      fetchButton.disabled = false;
    }
  }

  // Audio broadcast functionality
  function broadcastIntelligence() {
    if (speechSynth.speaking) {
      speechSynth.cancel();
    }

    const textToRead = extractionResult.textContent;
    if (
      !textToRead ||
      textToRead.includes("NO INTELLIGENCE RECEIVED") ||
      textToRead.includes("ERROR:")
    ) {
      document.getElementById("status-message").textContent =
        "ERROR: No valid intelligence to broadcast";
      return;
    }

    currentUtterance = new SpeechSynthesisUtterance(textToRead);

    // Set selected voice
    const selectedOption = voiceSelect.selectedOptions[0];
    if (selectedOption) {
      const voices = speechSynth.getVoices();
      for (let i = 0; i < voices.length; i++) {
        if (voices[i].name === selectedOption.getAttribute("data-name")) {
          currentUtterance.voice = voices[i];
          break;
        }
      }
    }

    // Set speech rate
    currentUtterance.rate = parseFloat(rateInput.value);

    // Update UI during speech
    currentUtterance.onstart = function () {
      statusText.textContent = "BROADCASTING";
      playBtn.disabled = true;
      pauseBtn.disabled = false;
      stopBtn.disabled = false;
      document.getElementById("status-message").textContent =
        "Transmission in progress...";
    };

    currentUtterance.onend = function () {
      statusText.textContent = "BROADCAST COMPLETE";
      playBtn.disabled = false;
      pauseBtn.disabled = true;
      stopBtn.disabled = true;
      document.getElementById("status-message").textContent =
        "Transmission complete";
    };

    currentUtterance.onerror = function (event) {
      console.error("SpeechSynthesis error:", event);
      document.getElementById("status-message").textContent =
        "ERROR: Transmission failed";
      statusText.textContent = "BROADCAST ERROR";
      playBtn.disabled = false;
      pauseBtn.disabled = true;
      stopBtn.disabled = true;
    };

    speechSynth.speak(currentUtterance);
  }

  // Pause speech
  function pauseSpeech() {
    if (speechSynth.speaking) {
      if (speechSynth.paused) {
        speechSynth.resume();
        pauseBtn.textContent = "PAUSE TRANSMISSION";
        document.getElementById("status-message").textContent =
          "Transmission resumed";
        statusText.textContent = "BROADCASTING";
      } else {
        speechSynth.pause();
        pauseBtn.textContent = "RESUME TRANSMISSION";
        document.getElementById("status-message").textContent =
          "Transmission paused";
        statusText.textContent = "BROADCAST PAUSED";
      }
    }
  }

  // Stop speech
  function stopSpeech() {
    speechSynth.cancel();
    statusText.textContent = "BROADCAST ABORTED";
    document.getElementById("status-message").textContent =
      "Transmission aborted";
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    pauseBtn.textContent = "PAUSE TRANSMISSION";
  }

  // Event listeners
  fetchButton.addEventListener("click", fetchFiles);
  playBtn.addEventListener("click", broadcastIntelligence);
  pauseBtn.addEventListener("click", pauseSpeech);
  stopBtn.addEventListener("click", stopSpeech);

  // File upload handling
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("file-input");
  const selectFileBtn = document.getElementById("select-file-btn");
  const processBtn = document.getElementById("process-btn");
  const clearBtn = document.getElementById("clear-btn");
  const progressContainer = document.getElementById("progress-container");
  const progressBar = document.getElementById("progress-bar");
  const loader = document.getElementById("loader");

  // Handle file selection
  selectFileBtn.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", handleFileSelect);

  // Drag and drop events
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false);
  });

  function highlight() {
    dropArea.classList.add("highlight");
  }

  function unhighlight() {
    dropArea.classList.remove("highlight");
  }

  dropArea.addEventListener("drop", handleDrop, false);

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) {
      handleFiles(files);
    }
  }

  function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length) {
      handleFiles(files);
    }
  }

  function handleFiles(files) {
    const file = files[0];
    if (!file.type.match("image.*")) {
      alert("ERROR: Only image files are supported!");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      document.getElementById("preview-container").style.display = "block";
      document.getElementById("drop-area").style.display = "none";

      statusText.textContent = "IMAGE ACQUIRED";
    };
    reader.readAsDataURL(file);
  }

  // Process image with Tesseract OCR
  processBtn.addEventListener("click", processImage);

  async function processImage() {
    try {
      statusText.textContent = "EXTRACTING DATA";
      progressContainer.style.display = "block";
      loader.style.display = "block";
      processBtn.disabled = true;
      clearBtn.disabled = true;

      const result = await Tesseract.recognize(imagePreview.src, "eng", {
        logger: (progress) => {
          if (progress.status === "recognizing text") {
            const progressPercent = progress.progress * 100;
            progressBar.style.width = `${progressPercent}%`;
          }
        },
      });

      // Display OCR result
      const extractedText = result.data.text;
      extractionResult.innerHTML = `<div class="intel-container">
          <h3 class="intel-headline">EXTRACTED DOCUMENT TEXT</h3>
          <p class="intel-content">${extractedText}</p>
        </div>`;

      // Enable audio controls
      playBtn.disabled = false;

      statusText.textContent = "EXTRACTION COMPLETE";
      document.getElementById("current-status").classList.add("status-success");
    } catch (error) {
      console.error("OCR Error:", error);
      extractionResult.innerHTML = `<p class="error-message">ERROR: OCR processing failed.<br>Details: ${error.message}</p>`;
      statusText.textContent = "EXTRACTION FAILED";
      document.getElementById("current-status").classList.add("status-error");
    } finally {
      progressContainer.style.display = "none";
      loader.style.display = "none";
      processBtn.disabled = false;
      clearBtn.disabled = false;
    }
  }

  // Clear uploaded data
  clearBtn.addEventListener("click", () => {
    imagePreview.src = "";
    document.getElementById("preview-container").style.display = "none";
    document.getElementById("drop-area").style.display = "block";
    extractionResult.innerHTML =
      "NO INTELLIGENCE RECEIVED. AWAITING DATA UPLOAD.";
    fileInput.value = "";
    progressBar.style.width = "0%";
    playBtn.disabled = true;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    statusText.textContent = "AWAITING MISSION";
    document
      .getElementById("current-status")
      .classList.remove("status-success", "status-error");
    document.getElementById("status-message").textContent = "";
  });
});
