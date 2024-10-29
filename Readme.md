
## Overview

This provides a comprehensive guide to setting up and using a satellite image processing and pattern extraction application using Flask, OpenCV, and other relevant libraries.

## Installation Guide

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

### Step 1: Clone the Repository

Start by cloning the repository from GitHub or your preferred source control:

```bash
git clone https://github.com/itsAPK/satellite-pattern-extraction.git
cd satellite-pattern-extraction
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

Creating a virtual environment helps manage dependencies specific to this project:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Step 3: Install Required Packages

Install the necessary Python packages using pip. A `requirements.txt` file should be included in the repository:

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, manually install the required libraries:

```bash
pip install Flask opencv-python numpy Pillow rasterio openpyxl scikit-learn kneed pandas
```

### Step 4: Setup Upload Directory

The application requires a directory to store uploaded images. This is automatically created in the code, but you can verify it exists:

```python
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
```

### Step 5: Run the Application

To start the Flask application, run the following command in your terminal:

```bash
python app.py
```

The application will be accessible at `http://127.0.0.1:5000/`.

## Application Workflow

### Image Upload and Processing

1. **Upload an Image**:
   - Users can upload TIFF or JPEG images through the web interface.
   - The application checks if the file is valid and saves it to the specified upload directory.

2. **Extract Geolocation**:
   - The application attempts to extract latitude and longitude from both image metadata and EXIF data.
   - If the image is a TIFF file, it is converted to JPEG for further processing.

3. **EXIF Data Extraction**:
   - All relevant EXIF data is extracted and prepared for JSON serialization.
   - This includes various metadata fields except GPS-related tags.

4. **Color Analysis**:
   - Users can specify a region of interest (ROI) within the uploaded image.
   - The application processes this ROI to analyze colors using KMeans clustering.
   - It generates stencil images for dominant colors found in the ROI.

5. **Excel Reporting**:
   - The results of color analysis and metadata are compiled into an Excel file with multiple sheets.
   - The main data sheet includes average colors, area measurements, and geolocation data.
   - A separate sheet lists dominant colors along with their representations.

### Key Functions Explained

1. **get_lat_lon_from_image(image_path)**:
   - Extracts latitude and longitude from satellite image metadata using `rasterio`.

2. **get_lat_lon_from_exif(image_path)**:
   - Retrieves latitude and longitude from EXIF GPS data if available.

3. **serialize_exif_data(exif_info)**:
   - Prepares EXIF data for JSON serialization by converting fractions to floats and excluding byte values.

4. **compress_image(image_path)**:
   - Compresses uploaded images to reduce file size while maintaining quality.

5. **extract_patterns()**:
   - Handles requests for color analysis on specified ROIs, processes images, and returns analysis results in JSON format.

6. **analyze_colors(roi_image)**:
   - Uses KMeans clustering to identify dominant colors in the ROI and generates stencil images for each color.

7. **preprocess_roi(roi_image)**:
   - Applies median blur and sharpening filters to enhance the ROI before color analysis.

8. **rgb_to_hex(rgb)**:
   - Converts RGB color values into HEX format for easier representation in reports.

## Conclusion

This Flask application provides a robust framework for image processing tasks, including geolocation extraction and color analysis. By following this documentation, users can successfully set up, run, and utilize the application for their image processing needs. For any issues or contributions, users are encouraged to refer to the repository's issue tracker or submit pull requests as needed.

