document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const themeToggle = document.getElementById("theme-toggle");
    const loading = document.getElementById("loading");
    const resultSection = document.getElementById("result-section");

    // Dark Mode Logic
    if (localStorage.getItem("theme") === "dark") document.body.setAttribute("data-theme", "dark");
    themeToggle.addEventListener("click", () => {
        const isDark = document.body.getAttribute("data-theme") === "dark";
        document.body.setAttribute("data-theme", isDark ? "light" : "dark");
        localStorage.setItem("theme", isDark ? "light" : "dark");
    });

    // Drag & Drop Logic
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.style.borderColor = "#28a745"; });
    dropZone.addEventListener("dragleave", () => dropZone.style.borderColor = "var(--primary-color)");
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--primary-color)";
        if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) handleUpload(e.target.files[0]);
    });

    document.querySelectorAll(".sample-card").forEach((sampleCard) => {
        sampleCard.addEventListener("click", async () => {
            sampleCard.disabled = true;
            try {
                const response = await fetch(sampleCard.dataset.sampleUrl);
                if (!response.ok) throw new Error("Unable to load the sample image.");

                const image = await response.blob();
                const filename = decodeURIComponent(sampleCard.dataset.sampleUrl.split("/").pop());
                handleUpload(new File([image], filename, { type: image.type }));
            } catch (error) {
                alert("Unable to load the sample image. Please try again.");
            } finally {
                sampleCard.disabled = false;
            }
        });
    });

    let currentResult = {};

    function handleUpload(file) {
        if (!['image/jpeg', 'image/png'].includes(file.type)) {
            alert("Please upload a valid JPG or PNG blood smear image.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        resultSection.classList.add("hidden");
        loading.classList.remove("hidden");

        fetch("/predict", { method: "POST", body: formData })
            .then(res => res.json())
            .then(data => {
                loading.classList.add("hidden");
                if (data.error) return alert(data.error);
                
                currentResult = data;
                displayResult(data);
                saveToHistory(data);
            })
            .catch(err => {
                loading.classList.add("hidden");
                alert("An error occurred during prediction.");
            });
    }

    function displayResult(data) {
        const predText = document.getElementById("prediction-text");
        const interpretationElement = document.getElementById("interpretation-text");
        const confFill = document.getElementById("confidence-fill");
        
        // 1. Hide the disease name if confidence is low (< 70%)
        if (data.confidence >= 70) {
            predText.textContent = `Diagnosis: ${data.result}`;
            predText.style.color = data.result === "Parasitized (Malaria Detected)" ? "var(--danger)" : "var(--success)";
        } else {
            // Hide the actual result and show a neutral message
            predText.textContent = `Diagnosis: Inconclusive`;
            predText.style.color = "var(--danger)"; // Using red/danger color to indicate attention is needed
        }
        
        // 2. Update the confidence progress bar
        confFill.style.width = `${data.confidence}%`;
        confFill.style.backgroundColor = data.confidence >= 90 ? "var(--success)" : (data.confidence >= 70 ? "var(--warning)" : "var(--danger)");
        
        document.getElementById("confidence-text").textContent = `Confidence: ${data.confidence}%`;
        
        // 3. Professional Interpretation Messages
        if (data.confidence >= 90) {
            interpretationElement.innerHTML = "<strong>High Confidence:</strong> The AI is highly certain of this analysis.";
            interpretationElement.className = "interpretation text-success";
        } else if (data.confidence >= 70) {
            interpretationElement.innerHTML = "<strong>Medium Confidence:</strong> The model is moderately certain. Clinical correlation is recommended.";
            interpretationElement.className = "interpretation text-warning";
        } else {
            interpretationElement.innerHTML = "⚠️ <strong>Analysis Withheld:</strong> The AI confidence is too low to provide a safe diagnosis. We strongly advise manual verification by a certified pathologist, or please upload a clearer, well-lit, and tightly focused blood smear image.";
            interpretationElement.className = "interpretation text-danger";
        }

        // 4. Show the images
        document.getElementById("original-img").src = data.original_image;
        document.getElementById("gradcam-img").src = data.gradcam_image;
        
        // Display the result section
        resultSection.classList.remove("hidden");
    }

    // PDF Download
    document.getElementById("download-btn").addEventListener("click", () => {
        fetch("/download_report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(currentResult)
        })
        .then(res => res.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "Malaria_Report.pdf";
            a.click();
        });
    });

    // History Dashboard (Local Storage)
    function saveToHistory(data) {
        let history = JSON.parse(localStorage.getItem("malariaHistory") || "[]");
        history.unshift(data);
        if (history.length > 5) history.pop();
        localStorage.setItem("malariaHistory", JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        const historyContainer = document.getElementById("history-container");
        let history = JSON.parse(localStorage.getItem("malariaHistory") || "[]");
        historyContainer.innerHTML = history.map(item => `
            <div class="history-item">
                <img src="${item.original_image}" alt="Scan">
                <strong>${item.result}</strong><br>
                <small>${item.confidence}%</small>
            </div>
        `).join("");
    }
    renderHistory();
});