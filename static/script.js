function displayFileName() {
    const input = document.getElementById('imageInput');
    const fileLabel = document.getElementById('custom-file-label');
    const imagePreview = document.getElementById('imagePreview');

    if (input.files && input.files[0]) {
        fileLabel.textContent = input.files[0].name;

        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.innerHTML = `<img src="${e.target.result}" class="preview-image" alt="Preview">`;
        };
        reader.readAsDataURL(input.files[0]);
    } else {
        fileLabel.textContent = 'No file chosen';
        imagePreview.innerHTML = '';
    }
}

const dropArea = document.getElementById('drop-area');
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
});
['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
});

dropArea.addEventListener('drop', function (e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) {
        document.getElementById('imageInput').files = files;
        displayFileName();
    }
});

async function uploadAndPredict() {
    const input = document.getElementById('imageInput');
    const resultCard = document.getElementById('resultCard');
    const result = document.getElementById('result');
    const file = input.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    result.innerHTML = '<div class="d-flex align-items-center"><div class="spinner-border spinner-border-sm text-primary me-2" role="status"></div> Analyzing image...</div>';
    resultCard.style.display = 'block';

    const reader = new FileReader();
    reader.onloadend = async function () {
        const base64Image = reader.result;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: base64Image })
            });

            const data = await response.json();
            result.textContent = "Prediction: " + data.predictio;
        } catch (error) {
            result.textContent = "Error occurred while predicting.";
            console.error(error);
        }
    };

    reader.readAsDataURL(file);
}
