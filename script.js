document.addEventListener('DOMContentLoaded', () => {
    // Get all the important HTML elements
    const uploadForm = document.getElementById('upload-form');
    const image1Input = document.getElementById('image1');
    const image2Input = document.getElementById('image2');
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');
    const modelSelect = document.getElementById('model-select');
    
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    const loader = document.getElementById('loader');
    
    const scoreEl = document.getElementById('score');
    const scoreDescriptionEl = document.getElementById('score-description');
    const heatmapImageEl = document.getElementById('heatmap-image');

    // Function to show a preview of the uploaded image
    function showPreview(input, preview) {
        input.addEventListener('change', () => {
            const file = input.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
    }

    showPreview(image1Input, preview1);
    showPreview(image2Input, preview2);

    // This runs when you click the "Compare Images" button
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!image1Input.files[0] || !image2Input.files[0]) {
            alert('Please upload both images.');
            return;
        }

        // Show the loader and hide old results
        resultsContainer.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        loader.classList.remove('hidden');

        // Create a FormData object to send
        const formData = new FormData();
        formData.append('image1', image1Input.files[0]);
        formData.append('image2', image2Input.files[0]);
        // Add the selected model from the dropdown
        formData.append('model', modelSelect.value);

        try {
            // Send the data to the Flask server
            const response = await fetch('http://127.0.0.1:5000/compare', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Server error');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            resultsContent.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        } finally {
            // Hide the loader and show the new results
            loader.classList.add('hidden');
            resultsContent.classList.remove('hidden');
        }
    });

    // This function puts the results on the page
    function displayResults(data) {
        const scorePercent = (data.similarity_score * 100).toFixed(2);
        scoreEl.textContent = `${scorePercent}%`;

        // This description changes based on the model used
        if (data.model_used === 'ssim') {
            scoreDescriptionEl.textContent = 'This SSIM score measures the structural similarity. A score of 100% means the images are perceptually identical.';
        } else {
            // This works for VGG16, ResNet50, and MobileNetV2
            let modelName = data.model_used.toUpperCase();
            scoreDescriptionEl.textContent = `This ${modelName} (Deep Learning) score measures the semantic similarity. A score of 100% means the images are contextually and conceptually identical.`;
        }

        // Display the heatmap
        heatmapImageEl.src = data.heatmap_image;
    }
});