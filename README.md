# Dl_hackathon
# Eye Vessel Segmentation and Health Indicator Web Application

## Overview

This web application processes images of the eye to segment the retinal vessels and provides a potential health indicator based on the vessel structure. It uses a deep learning model (U-Net with ResNet34 backbone) trained for vessel segmentation, followed by feature extraction from the segmented vessels and clustering using a pre-trained KMeans model.

The application performs the following steps:

1.  **Image Upload:** Accepts an image file through a web interface or an API endpoint.
2.  **Region of Interest (ROI) Detection (Optional):** Attempts to detect the eye region in the uploaded image using a Haar Cascade classifier. If an eye is detected, the segmentation is focused on this ROI.
3.  **Vessel Segmentation:** Uses a pre-trained fastai U-Net model to segment the retinal vessels in the input image (or the detected ROI).
4.  **Post-processing:** Applies morphological operations (connected components analysis and closing) to refine the segmentation mask.
5.  **Feature Extraction:** Extracts quantitative features from the segmented binary mask, such as density, branch points, endpoints, tortuosity, vein length, and vein width.
6.  **Feature Normalization:** Normalizes the extracted features using a pre-trained MinMaxScaler.
7.  **Health Indicator Prediction:** Uses a pre-trained KMeans clustering model to classify the normalized features into predefined health-related clusters.
8.  **Output:** Returns the segmented vessel mask as a Base64 encoded PNG image and the predicted health indicator text as a JSON response.

## Prerequisites

Before running the application, ensure you have the following installed:

* **Git** (for cloning the repository)
* **Python 3.x**
* **pip** (Python package installer)
* **Required Python libraries:** You can install them using the `requirements.txt` file (create one with the following content):

    ```
    fastai
    torch
    torchvision
    opencv-python
    scikit-learn
    joblib
    Pillow
    Flask
    Flask-CORS
    scipy
    scikit-image
    numpy
    ```
* **Pre-trained models and other necessary files:**
    * `bestmodel.pth`: The weights file for the vessel segmentation U-Net model.
    * `scaler.joblib`: The pre-trained MinMaxScaler model for feature normalization.
    * `kmeans.joblib`: The pre-trained KMeans clustering model.
    * `haarcascade_eye.xml`: The Haar Cascade XML file for eye detection.

## Setup

1.  **Clone the repository:** If the code is hosted on a Git platform (like GitHub), clone it using:

    ```bash
    git clone <repository_url>
    ```

2.  **Navigate to the project directory:** Change your current directory to the `dl/dlhackathon` folder:

    ```bash
    cd dl/dlhackathon
    ```

3.  **Verify the project directory structure:** Ensure your `dl/dlhackathon` folder contains the following files and directories:

    ```
    dl/dlhackathon/
    ├── dummy_data/         (Optional: Used for model loading, can be empty)
    ├── models/             (Contains: bestmodel.pth, scaler.joblib, kmeans.joblib)
    ├── __pycache__/        (Python cache directory, usually created automatically)
    ├── templates/          (Contains: index.html - the web interface file)
    ├── app.py              (The main Flask application file)
    ├── custom_components.py (Contains custom classes and functions)
    ├── haarcascade_eye.xml (Haar Cascade file for eye detection)
    ├── index.html          (Potentially another HTML file, ensure the correct one is used if different)
    ├── Procfile            (Optional: For deployment on platforms like Heroku)
    ├── requirements.txt    (Lists Python dependencies)
    ├── script.js           (Optional: JavaScript file for the web interface)
    └── style.css           (Optional: CSS file for the web interface)
    ```

4.  **Place the pre-trained model files** (`bestmodel.pth`, `scaler.joblib`, `kmeans.joblib`) inside the `models` directory.
5.  **Place the `haarcascade_eye.xml` file** in the root directory of the `dl/dlhackathon` folder.
6.  The web interface (`index.html`) is expected to be in the `templates` directory.

## Running the Application

1.  **Install the required Python libraries:** Navigate to the `dl/dlhackathon` directory in your terminal and run:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Flask development server:** In the same `dl/dlhackathon` directory, execute:

    ```bash
    python app.py
    ```

3.  The server will start, and you should see output similar to:

    ```
    Starting Flask server on port 5001...
     * Serving Flask app 'app'
     * Debug mode: on
    ...
    ```

4.  **Open the website:** Open your web browser and go to `http://127.0.0.1:5002/` (or the address and port shown in your terminal). You should see the web interface provided by `templates/index.html`.

5.  **Upload an image:** Use the file input field to select an eye image and click "Analyze". The results (health indicator and segmented vessel image) will be displayed on the page.

## API Endpoint

The application also provides an API endpoint for making predictions:

* **Endpoint:** `/predict`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`
* **Request Body:** Should include a file named `image` containing the eye image.

The API will return a JSON response with the following structure:

```json
{
  "mask_image_base64": "...",      // Base64 encoded PNG image of the segmented vessels
  "predicted_cluster": 0,           // Integer representing the predicted cluster ID (if clustering models are loaded)
  "health_indicator_text": "Fatigue" // Textual description of the health indicator (if clustering models are loaded)
}
