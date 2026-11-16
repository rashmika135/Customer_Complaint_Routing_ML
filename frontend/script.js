const form = document.getElementById("complaint-form");
const complaintInput = document.getElementById("complaint");
const characterCount = document.getElementById("character-count");
const submitButton = document.getElementById("submit-btn");
const buttonText = document.getElementById("button-text");
const errorMessage = document.getElementById("error-message");

const emptyResult = document.getElementById("empty-result");
const predictionResult = document.getElementById("prediction-result");
const categoryText = document.getElementById("predicted-category");
const confidenceValue = document.getElementById("confidence-value");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceProgress = document.getElementById("confidence-progress");
const routingStatus = document.getElementById("routing-status");
const routingDescription = document.getElementById("routing-description");

function updateCharacterCount() {
    characterCount.textContent = `${complaintInput.value.length} / 10,000`;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = !message;
}

function setLoading(loading) {
    submitButton.disabled = loading;
    buttonText.textContent = loading ? "Classifying..." : "Classify complaint";
}

function showResult(data) {
    const confidence = Number(data.confidence);
    const percentage = Math.round(confidence * 100);
    const needsReview = data.status === "Needs Review";

    categoryText.textContent = data.predicted_category;
    confidenceValue.textContent = `${percentage}%`;
    confidenceBar.style.width = `${percentage}%`;
    confidenceProgress.setAttribute("aria-valuenow", percentage);

    routingStatus.textContent = data.status;
    routingStatus.className = needsReview ? "status-pill review" : "status-pill auto";
    routingDescription.textContent = needsReview
        ? "The model is less confident about this complaint. Manual review is recommended."
        : "The model is confident enough to suggest automatic routing. No department assignment has been made.";

    emptyResult.hidden = true;
    predictionResult.hidden = false;
}

complaintInput.addEventListener("input", () => {
    updateCharacterCount();
    showError("");
});

document.getElementById("example-btn").addEventListener("click", () => {
    complaintInput.value = "I was charged twice for the same transaction and the bank has not refunded my money.";
    updateCharacterCount();
    showError("");
    complaintInput.focus();
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const complaint = complaintInput.value.trim();

    if (!complaint) {
        showError("Please enter a complaint first.");
        return;
    }

    showError("");
    setLoading(true);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ complaint })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(typeof data.detail === "string" ? data.detail : "Could not classify the complaint.");
        }

        if (typeof data.predicted_category !== "string" ||
            typeof data.status !== "string" ||
            !Number.isFinite(Number(data.confidence))) {
            throw new Error("The server returned an unexpected result.");
        }

        showResult(data);
    } catch (error) {
        showError(error.message || "Something went wrong. Please try again.");
    } finally {
        setLoading(false);
    }
});

updateCharacterCount();
