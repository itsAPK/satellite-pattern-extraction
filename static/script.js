document.getElementById('upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const imgElement = document.getElementById('uploaded-image');
        imgElement.src = e.target.result;
 
        // Initialize canvas for ROI selection
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        
        imgElement.onload = function() {
            // Set canvas size based on image dimensions
            const imgWidth = imgElement.naturalWidth;
            const imgHeight = imgElement.naturalHeight;
 
            // Set maximum dimensions for canvas
            const maxWidth = window.innerWidth * 0.8; // 80% of window width
            const maxHeight = window.innerHeight * 0.8; // 80% of window height
 
            // Calculate scale to maintain aspect ratio
            let scaleFactor = Math.min(maxWidth / imgWidth, maxHeight / imgHeight);
            
            // Set canvas dimensions
            canvas.width = imgWidth * scaleFactor;
            canvas.height = imgHeight * scaleFactor;
 
            // Draw the image on the canvas
            context.drawImage(imgElement, 0, 0, imgWidth * scaleFactor, imgHeight * scaleFactor);
        };
    };
    
    if (file) {
        reader.readAsDataURL(file);
        
        // Upload the image to get latitude and longitude
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('latitude').innerText = data.latitude ? data.latitude.toFixed(6) : "N/A";
            document.getElementById('longitude').innerText = data.longitude ? data.longitude.toFixed(6) : "N/A";
              // Display EXIF Data
              const exifData = JSON.stringify(data.exif_data, null, 2); // Pretty-print JSON
              document.getElementById('exif-data').innerText = exifData; // Display EXIF data in <pre> element
        });
    }
});

// Allow user to select ROI on canvas
let isDrawing = false;
let startX, startY;

const canvas = document.getElementById('canvas');
canvas.addEventListener('mousedown', function(e) {
    isDrawing = true;
    startX = e.offsetX;
    startY = e.offsetY;
});

canvas.addEventListener('mousemove', function(e) {
    if (!isDrawing) return;

    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(document.getElementById('uploaded-image'), 0, 0);

    const width = e.offsetX - startX;
    const height = e.offsetY - startY;

    context.strokeStyle = 'red';
    context.strokeRect(startX, startY, width, height);
});

canvas.addEventListener('mouseup', function(e) {
    isDrawing = false;

    let width = e.offsetX - startX;
    let height = e.offsetY - startY;

    // Ensure width and height are positive
    if (width < 0) {
        startX += width; // Adjust starting point
        width *= -1;     // Make width positive
    }
    
    if (height < 0) {
        startY += height; // Adjust starting point
        height *= -1;     // Make height positive
    }

    // Display selected ROI coordinates
    document.getElementById('roi-coordinates').innerText =
        `(${startX}, ${startY}, ${width}, ${height})`;

    // Prepare data to send to the server for extraction
    const roiData = { x: Math.round(startX), y: Math.round(startY), width: Math.round(width), height: Math.round(height) };

    
    document.getElementById('extract-patterns').onclick = function() {
        fetch('/extract_patterns', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                filename: document.getElementById('upload').files[0].name,
                roi: roiData,
                latitude: document.getElementById('latitude').innerText,
                longitude: document.getElementById('longitude').innerText 
            })
        })
        .then(response => response.json())
        .then(data => {
            // Display average color
            document.getElementById('average-color').innerText =
                `RGB(${data.average_color.join(', ')})`;
            
            // Display area metrics
            document.getElementById('area-pixels').innerText =
                `${data.area_pixels} pixels`;
            document.getElementById('area-square-meters').innerText =
                `${data.area_square_meters.toFixed(2)} m²`;
            document.getElementById('length-meters').innerText =
                `${data.length_meters.toFixed(2)} m`;
    
            // Display unique colors with their images
            const colorsContainerDiv = document.createElement("div");
            colorsContainerDiv.id = "colors-container";
            colorsContainerDiv.innerHTML = "<h3>Colors:</h3>";
            
            data.dominant_colors.forEach(colorInfo => {
                const colorDiv = document.createElement("div");
                colorDiv.style.display = "flex";
                colorDiv.style.alignItems = "center";
    
                const colorImagePath = document.createElement("div");
                colorImagePath.style.background = colorInfo.hex; // Assuming hex is used as a placeholder here for color display;
                colorImagePath.style.width = "50px"; // Set size for display purposes
                colorImagePath.style.height = "50px"; // Set size for display purposes
                colorImagePath.style.margin = "5px";
    
                const rgbText = document.createElement("span");
                rgbText.innerText = `RGB(${colorInfo.rgb.join(', ')}) - ${colorInfo.hex}`;
                
                colorDiv.appendChild(colorImagePath);
                colorDiv.appendChild(rgbText);
                
                colorsContainerDiv.appendChild(colorDiv);
            });
    
            // Append colors container to body or a specific div in your HTML structure.
            document.body.appendChild(colorsContainerDiv);
    
            // Display extracted ROI image
            const roiImagePath = data.roi_image_path; // Get ROI image path from response
            const roiImageTag = document.createElement("img");
            roiImageTag.src = roiImagePath;
            roiImageTag.style.width = "400px"; // Set size for display purposes
            
            document.body.appendChild(roiImageTag); // Append it to body or a specific div in your HTML structure.
    
          
        });
    };
});

// Zoom functionality
let scaleFactor = 1.0; // Initial scale factor

document.getElementById('zoom-in').addEventListener('click', function() {
     scaleFactor += 0.1; // Increase scale factor by 10%
     updateCanvasScale();
});

document.getElementById('zoom-out').addEventListener('click', function() {
     scaleFactor -= 0.1; // Decrease scale factor by 10%
     if (scaleFactor <= 0.1) scaleFactor = 0.1; // Prevent negative or zero scaling
     updateCanvasScale();
});

function updateCanvasScale() {
     const imgElement = document.getElementById('uploaded-image');
     const canvas = document.getElementById('canvas');
     
     // Clear and resize canvas based on new scale
     canvas.width = imgElement.naturalWidth * scaleFactor;
     canvas.height = imgElement.naturalHeight * scaleFactor;

     const context = canvas.getContext('2d');
     
     // Draw scaled image on canvas
     context.clearRect(0, 0, canvas.width, canvas.height);
     context.drawImage(imgElement, 0, 0, imgElement.naturalWidth * scaleFactor, imgElement.naturalHeight * scaleFactor);
}


