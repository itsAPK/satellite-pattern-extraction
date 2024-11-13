import io
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from fractions import Fraction
import base64
from sklearn.cluster import KMeans, MiniBatchKMeans
from werkzeug.utils import secure_filename
from sklearn.utils import shuffle
import pandas as pd
from openpyxl.drawing.image import Image as OpenPyXLImage
from openpyxl.styles import PatternFill

app = Flask(__name__)


# Define a directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define a scale for conversion (pixels per meter)
PIXELS_PER_METER = 10  # Adjust based on your specific image scale

import rasterio

def get_lat_lon_from_image(image_path):
    """Extract latitude and longitude from satellite image metadata."""
    try:
        with rasterio.open(image_path) as src:
            # Get the bounding box of the image in geographic coordinates
            bounds = src.bounds
            # Calculate the center point of the bounding box
            lat = (bounds.top + bounds.bottom) / 2
            lon = (bounds.left + bounds.right) / 2
            
            return lat, lon
    except Exception as e:
        print(f"Error extracting coordinates from image: {e}")
    
    return None, None

def convert_ifdrational(value):
    """Convert IFDRational or similar types to float."""
    if isinstance(value, (tuple, list)):
        return [convert_ifdrational(v) for v in value]
    elif isinstance(value, Fraction):
        return float(value)
    elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
        return float(value.numerator) / float(value.denominator)
    return value

def get_lat_lon_from_exif(image_path):
    """Extract latitude and longitude from image EXIF data."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None, None

        lat = lon = None
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                for key in value:
                    sub_tag = GPSTAGS.get(key, key)
                    if sub_tag == 'GPSLatitude':
                        lat = value[key]
                    elif sub_tag == 'GPSLongitude':
                        lon = value[key]
                break

        if lat is not None and lon is not None:
            lat_decimal = lat[0] + lat[1] / 60 + lat[2] / 3600
            lon_decimal = lon[0] + lon[1] / 60 + lon[2] / 3600
            
            # Handle direction indicators
            if exif_data.get(1) == 'S':  # Latitude South
                lat_decimal = -lat_decimal
            if exif_data.get(3) == 'W':  # Longitude West
                lon_decimal = -lon_decimal

            return convert_ifdrational(lat_decimal), convert_ifdrational(lon_decimal)
        
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
    
    return None, None

def serialize_exif_data(exif_info):
    """Serialize EXIF data for JSON response, ensuring no bytes are included."""
    serialized_exif_data = {}
    for key, value in exif_info.items():
        if isinstance(value, (list, tuple)):
            serialized_exif_data[key] = [convert_ifdrational(v) for v in value if not isinstance(v, bytes)]
        elif isinstance(value, Fraction):
            serialized_exif_data[key] = float(value)
        elif isinstance(value, bytes):  # Skip bytes
              pass
    return serialized_exif_data

Image.MAX_IMAGE_PIXELS = None  # Disable the limit (use cautiously)


def convert_tiff_to_jpeg(tiff_path):
    jpeg_path = tiff_path.replace('.tiff', '.jpeg').replace('.tif', '.jpeg')  # Change the extension as needed
    with Image.open(tiff_path) as img:
        img.convert('RGB').save(jpeg_path, 'JPEG')
    return jpeg_path

@app.route('/')
def index():
    return render_template('index.html')

# Function to compress image
def compress_image(image_path, max_size=(1024, 1024), quality=90):
    """Compress the image to the specified max_size and quality."""
    with Image.open(image_path) as img:
        # Resize the image
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save to a BytesIO object
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality)
        img_byte_arr.seek(0)

        # Save the compressed image back to a file
        compressed_image_path = image_path.replace('.tiff', '_compressed.jpeg').replace('.tif', '_compressed.jpeg')
        with open(compressed_image_path, 'wb') as f:
            f.write(img_byte_arr.getvalue())
    
    return compressed_image_path

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Sanitize filename and check file type
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(filepath)
        m = filepath

        # Check file size
       
        latitude, longitude = get_lat_lon_from_image(filepath)

        response_filename = filename
        # Check if the uploaded file is a TIFF
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            # Convert TIFF to JPEG and keep the path
            jpeg_path = convert_tiff_to_jpeg(m)

            # Optionally, you can delete the original TIFF file if you no longer need it
            os.remove(m)
            m = jpeg_path  # Update filepath to the new JPEG file
            response_filename = filename.replace('.tiff', '.jpeg').replace('.tif', '.jpeg')


        # Get all EXIF data from the converted or original image
       # exif_data = get_exif_data(m)

        # Extract latitude and longitude if available

        # Prepare response
        return jsonify({
            'filename': response_filename,
            'latitude': str(latitude) if latitude is not None else None,
            'longitude': str(longitude) if longitude is not None else None,
            #**exif_data 
        }), 200
       
    except Exception as e:
        print(f"Error during file upload: {e}")
        return jsonify({'error': 'Failed to process the image.'}), 500
    
def get_exif_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif() or {}

        exif_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_info[tag] = convert_ifdrational(value)  # Convert any fractions to floats
            
        # Ensure the EXIF data is serializable to JSON
        serialized_exif_data = {}
        for key, value in exif_info.items():
            # Skip GPS-related tags (GPSInfo is usually a tag key for GPS data)
            if 'GPS' in key:
                continue  # Skip any key containing 'GPS'
            
            if isinstance(value, (list, tuple)):
                serialized_exif_data[key] = [
                    convert_ifdrational(v) if not isinstance(v, bytes) else v.decode('utf-8', errors='ignore') 
                    for v in value
                ]
            elif isinstance(value, Fraction):
                serialized_exif_data[key] = float(value)
            elif isinstance(value, bytes):
                serialized_exif_data[key] = value.decode('utf-8', errors='ignore')  # Convert bytes to string
            else:
                serialized_exif_data[key] = value
        
        print(serialized_exif_data)
                
        return serialized_exif_data

    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
    
    return {}


def rgb_to_hex(rgb):
    """Convert RGB color to HEX format."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])





@app.route('/extract_patterns', methods=['POST'])
def extract_patterns():
    data = request.json
    filename = data.get('filename')
    roi = data.get('roi')

    if not filename or not roi:
        return jsonify({'error': 'Missing filename or ROI data.'}), 400

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = cv2.imread(image_path)
    
    with rasterio.open(image_path) as src:
        i = src.read()  # Read the image data
        transform = src.transform  # Get the affine transform for coordinate conversion
        
        # Extract coordinates of the ROI corners
        x_min, y_min = transform * (roi['x'], roi['y'])
        x_max, y_max = transform * (roi['x'] + roi['width'], roi['y'] + roi['height'])

        # Create a list of corner coordinates (longitude, latitude)
        roi_corners = {
            'top_left': (x_min, y_max),
            'top_right': (x_max, y_max),
            'bottom_left': (x_min, y_min),
            'bottom_right': (x_max, y_min)
        }

    x, y, width, height = roi['x'], roi['y'], roi['width'], roi['height']
    
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        return jsonify({'error': 'ROI is out of bounds.'}), 400

    roi_image = image[y:y + height, x:x + width]
    
    # Apply Gaussian filter and Canny edge detection on the ROI
    gaussian_blurred = cv2.GaussianBlur(roi_image, (5, 5), 0)
    edges = cv2.Canny(gaussian_blurred, 100, 200)

    # Save processed images
    gaussian_image_path = os.path.join(UPLOAD_FOLDER, f'gaussian_{filename}')
    edges_image_path = os.path.join(UPLOAD_FOLDER, f'edges_{filename}')
    
    cv2.imwrite(gaussian_image_path, gaussian_blurred)
    cv2.imwrite(edges_image_path, edges)

    # Find contours from edges detected by Canny
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a blank image or the original ROI for visualization
    contour_image = np.zeros_like(roi_image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

    # Save the contour image
    contour_image_path = os.path.join(UPLOAD_FOLDER, f'contours_{filename}')
    cv2.imwrite(contour_image_path, contour_image)

    if roi_image.size > 0:
        roi_image_processed = preprocess_roi(roi_image)

        # Save the processed ROI image
        roi_image_filename = f'roi_{filename}'
        roi_image_path = os.path.join(UPLOAD_FOLDER, roi_image_filename)
        cv2.imwrite(roi_image_path, roi_image_processed)

        color_info, area_square_meters, avg_color_rgb, breadth, length, color_representation_image, common_color = analyze_colors(roi_image_processed)

        # Save stencil images and update paths
        dominant_colors = []
        for index, color in enumerate(color_info):
            
            dominant_colors.append({
                'Color' : '',
                'Hex': color['hex'],  # Use the correct key for background color
                'RGB': color['rgb'],
                'Stencil': '',
                'Path' : color['path'] # Use the saved path for stencil
            })

        # Prepare data for Excel
        excel_data = {
            'Average Color (RGB)': [avg_color_rgb],
            'Area (Pixels)': [width * height],
            'Area (Square Meters)': [area_square_meters],
            'Common Color': [common_color],
            'Latitude': [data.get('latitude')],
            'Longitude': [data.get('longitude')],
            'Breadth (Pixels)': [width],
            'Length (Pixels)': [height],
            'Breadth': [breadth],
            'Length': [length]
        }

        # Create a DataFrame for the main data
        main_df = pd.DataFrame(excel_data)

        # Create a DataFrame for dominant colors
        dominant_colors_df = pd.DataFrame(dominant_colors)

        # Save DataFrames to an Excel file
        excel_path = os.path.join(UPLOAD_FOLDER, f'patterns_data_{filename}.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            main_df.to_excel(writer, index=False, sheet_name='Main Data')
            dominant_colors_df.to_excel(writer, index=False, sheet_name='Dominant Colors')

            # Access the dominant colors sheet
            dominant_colors_sheet = writer.sheets['Dominant Colors']
            
            

            # Insert stencil images
            for index, color in enumerate(dominant_colors):
                print(color)
                fill_color = PatternFill(start_color=color['Hex'].replace('#', ''), end_color=color['Hex'].replace('#', ''), fill_type='solid')
                dominant_colors_sheet.cell(row=index + 2, column=1).fill = fill_color  # Start from row 2

                stencil_image_path = color['Path']  # Path to the stencil image
                if os.path.exists(stencil_image_path):  # Check if the file exists
                    try:
                        print('a')
                        img = OpenPyXLImage(stencil_image_path)
                        img.width = 100  # Adjust the image width if needed
                        img.height = 100  # Adjust the image height if needed
                        # Calculate the cell position (A5 is the starting point)
                        cell_row = 2 + index  # Starts from row 5 and increments
                        print(img)
                        dominant_colors_sheet.add_image(img, f'D{cell_row}')  # Insert image at A5, A6, etc.
                        print(f'Inserted image at D{cell_row}: {stencil_image_path}')
                        dominant_colors_sheet.row_dimensions[cell_row].height = 80  # Adjust as necessary

                    except Exception as e:
                        print(f'Error inserting image at A{cell_row}: {e}')
                else:
                    print(f'Image file not found: {stencil_image_path}')

                column_widths = {
            'A': 20,  # Width for Hex column
            'B': 20,  # Width for RGB column
            'C': 20,  # Width for Stencil column
            'D': 20, 
            'E' : 50  # Increased width for Image column to fit images
        }
        
                for column_letter, width in column_widths.items():
                     dominant_colors_sheet.column_dimensions[column_letter].width = width

        # Log the path of the generated Excel file
        print(f'Excel file created at: {excel_path}')

        return jsonify({
            'average_color': avg_color_rgb,
            'area_pixels': width * height,
            'area_square_meters': area_square_meters,
            'dominant_colors': color_info,
            'common_color': common_color,
            'roi_image_path': roi_image_path.replace('\\', '/'),
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'breadth_in_pixels': width,
            'length_in_pixels': height,
            'breadth': breadth,
            'length': length,
            'gaussian_image_path': gaussian_image_path.replace('\\', '/'),
            'edges_image_path': edges_image_path.replace('\\', '/'),
            'contour_image_path': contour_image_path.replace('\\', '/'),
            'excel_path': excel_path.replace('\\', '/') , # Return the path of the Excel file
            'roi_corners': roi_corners  # Return the corners of the ROI

        }), 200

def preprocess_roi(roi_image):
    """Preprocess the ROI: blur and sharpen."""
    roi_image_blurred = cv2.medianBlur(roi_image, 3)
    print("roi_image_blurred:" , roi_image_blurred)
    roi_image_resized = cv2.resize(roi_image_blurred, (roi_image.shape[1] * 2, roi_image.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    print("roi_image_resized:" , roi_image_resized)
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    print("sharpening_kernel:" , sharpening_kernel)
    return cv2.filter2D(roi_image_resized, -1, sharpening_kernel)


def process_contour(contour, roi_image, dominant_color, stencil_image):
    # Create mask for the contour
    contour_mask = np.zeros_like(roi_image, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Calculate mean color inside the contour
    mean_color = cv2.mean(roi_image, mask=contour_mask[:, :, 0])[:3]
    mean_color_int = np.array(mean_color).astype(int)

    # If the mean color inside the contour matches the dominant color, fill it
    if np.allclose(mean_color_int, dominant_color, atol=30):
        cv2.drawContours(stencil_image, [contour], -1, dominant_color.tolist(), thickness=cv2.FILLED)


def analyze_colors(roi_image):
    """Analyze colors in the ROI using KMeans clustering and return stencil images for each dominant color."""
    stencil_dir = UPLOAD_FOLDER + '/stencils'
    os.makedirs(stencil_dir, exist_ok=True)
    print(f"Stencil directory created at: {stencil_dir}")
    
    # Downsample the image to reduce the number of pixels
    downsample_factor = 8
    small_roi_image = cv2.resize(roi_image, (roi_image.shape[1] // downsample_factor, roi_image.shape[0] // downsample_factor))
    print(f"Downsampled image shape: {small_roi_image.shape}")
    
    # Reshape the image into a 2D array of pixels and shuffle them for clustering
    pixels = shuffle(small_roi_image.reshape(-1, 3))
    print(f"Pixels reshaped for clustering: {pixels.shape}")
    
    # Use MiniBatchKMeans for faster clustering
    kmeans = MiniBatchKMeans(n_clusters=4, init='k-means++', batch_size=1000)
    kmeans.fit(pixels)
    print(f"KMeans clustering completed. Cluster centers (dominant colors): {kmeans.cluster_centers_}")

    # Extract cluster centers as dominant colors
    dominant_colors_rgb = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    print(f"Dominant colors RGB: {dominant_colors_rgb}")
    
    # Prepare blank image for showing dominant color patterns
    filled_shapes_image = np.zeros_like(roi_image)
    
    # Convert to grayscale, apply Gaussian blur, and threshold for contour detection
    gray_roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    print(f"Converted ROI to grayscale.")
    
    blurred_roi_image = cv2.GaussianBlur(gray_roi_image, (3, 3), 0)
    print(f"Applied Gaussian blur.")
    
    _, thresh = cv2.threshold(blurred_roi_image, 127, 255, cv2.THRESH_BINARY)
    print(f"Threshold applied. Binary image shape: {thresh.shape}")
    
    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    print(f"Cleaned threshold image with morphological opening.")

    # Find contours on the binary image
    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found contours: {len(contours)}")

    color_info_list = []

    for i, dominant_color in enumerate(dominant_colors_rgb):
        print(f"Processing dominant color {i}: {dominant_color}")
        
        # Create mask for the current color based on KMeans labels
        mask = (labels == i).reshape(small_roi_image.shape[:2])
        mask_resized = cv2.resize(mask.astype(np.uint8), (roi_image.shape[1], roi_image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Create a blank image for drawing contours of this dominant color
        dominant_color_filled_image =np.zeros_like(roi_image, dtype=np.uint8)  # White background

        # Vectorized contour processing
        for contour in contours:
            # Create mask for the contour
            contour_mask = np.zeros_like(roi_image, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            # Calculate mean color inside the contour
            mean_color = cv2.mean(roi_image, mask=contour_mask[:, :, 0])[:3]
            mean_color_int = np.array(mean_color).astype(int)
            print(f"Mean color inside contour: {mean_color_int}")
            
            # If the mean color inside the contour matches the dominant color, fill it
            if np.allclose(mean_color_int, dominant_color, atol=30):  # Tolerance to account for color variations
                cv2.drawContours(dominant_color_filled_image, [contour], -1, dominant_color.tolist(), thickness=cv2.FILLED)

        # Save contour image for this dominant color
        stencil_image_path = os.path.join(stencil_dir, f'stencil_{i}.png')
        contour_pil = Image.fromarray(dominant_color_filled_image.astype('uint8'))
        
        try:
            contour_pil.save(stencil_image_path)
            print(f"Saved stencil image at: {stencil_image_path}")
        except Exception as e:
            print(f"Error saving stencil image: {e}")

        # Convert filled shapes image to base64 for JSON response (optional)
        buffered = io.BytesIO()
        contour_pil.save(buffered, format="PNG")
        filled_shapes_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Add color info to list
        color_info_list.append({
            'rgb': dominant_color.tolist(),
            'hex': rgb_to_hex(dominant_color),
            'stencil_image': filled_shapes_base64,
            'path': stencil_image_path.replace('\\', '/'),
            'contours': []
        })

    # Count occurrences of each label to find the most common color
    label_counts = np.bincount(labels)
    common_color_index = np.argmax(label_counts)
    common_color_rgb = dominant_colors_rgb[common_color_index]
    common_color_hex = rgb_to_hex(common_color_rgb)

    common_color = {
        'rgb': common_color_rgb.tolist(),
        'hex': common_color_hex
    }
    print(f"Common color found: {common_color}")

    # Calculate area and dimensions
    area_square_meters = (roi_image.shape[0] * roi_image.shape[1]) / (PIXELS_PER_METER ** 2)
    width = roi_image.shape[1] / PIXELS_PER_METER
    height = roi_image.shape[0] / PIXELS_PER_METER

    avg_color_rgb_intensity = np.mean(dominant_colors_rgb, axis=0).astype(int).tolist()
    print(f"Average RGB intensity: {avg_color_rgb_intensity}")

    return color_info_list, area_square_meters, avg_color_rgb_intensity, width, height, filled_shapes_image, common_color



def find_optimal_n_clusters(inertias):
    # Calculate the first derivative (difference)
    first_derivative = np.diff(inertias)
    
    # Calculate the second derivative (change of the first derivative)
    second_derivative = np.diff(first_derivative)
    
    # The optimal cluster is at the index where the second derivative is maximized
    # We can find the maximum point in the second derivative
    optimal_index = np.argmax(second_derivative) + 1  # +1 because np.diff reduces the size by 1
    
    # Ensure optimal_index doesn't exceed the bounds of the original clusters
    optimal_n_clusters = np.clip(optimal_index, 1, len(inertias))
    
    return optimal_n_clusters


if __name__ == '__main__':
    app.run(debug=True)