import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
import time
import json
import bcrypt
from typing import Optional, Tuple, Dict
import logging
import warnings
from numpy import dot
from numpy.linalg import norm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import mimetypes
import tempfile

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# Suppress TensorFlow and OpenCV Warnings
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("attendance_system.log"),
        logging.StreamHandler()
    ]
)

# -----------------------------
# Initialize Firebase
# -----------------------------
def initialize_firebase():
    """Initialize Firebase Admin SDK."""
    cred_path = "Code/ServiceAccountKey.json"  # Ensure this path is correct
    if not os.path.exists(cred_path):
        logging.critical(f"Service account key not found at {cred_path}. Please provide the correct path.")
        messagebox.showerror("Initialization Error", f"Service account key not found at {cred_path}. Please provide the correct path.")
        exit(1)
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://faceattendancerealtime-cfa63-default-rtdb.firebaseio.com/"
        })
        logging.info("Firebase initialized successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialize Firebase: {e}")
        messagebox.showerror("Initialization Error", f"Failed to initialize Firebase: {e}")
        exit(1)

initialize_firebase()

# -----------------------------
# Ensure Images Directory Exists
# -----------------------------
if not os.path.exists('Images'):
    os.makedirs('Images')
    logging.info("Created 'Images' directory.")

# -----------------------------
# Helper Functions for Password Hashing
# -----------------------------
def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a stored password against one provided by user."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# -----------------------------
# Shared State for Low Light Mode
# -----------------------------
shared_state = {
    'low_light': False
}

# -----------------------------
# Cosine Similarity Function
# -----------------------------
def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors."""
    return dot(a, b) / (norm(a) * norm(b))

# -----------------------------
# Constants
# -----------------------------
SIMILARITY_THRESHOLD = 0.5  # Initial threshold; can be adjusted dynamically

# -----------------------------
# Preprocess Image Function
# -----------------------------
def preprocess_image(img: np.ndarray, low_light: bool = False) -> np.ndarray:
    """
    Preprocess the image for better clarity and alignment.
    Args:
    img (np.ndarray): The original image captured from the webcam.
    low_light (bool): Flag indicating whether to apply low light preprocessing.
    Returns:
    np.ndarray: The preprocessed image.
    """
    try:
        if low_light:
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply Histogram Equalization
            img_equalized = cv2.equalizeHist(img_gray)
            # Convert back to BGR
            img = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2BGR)
            # Apply Gamma Correction
            gamma = 1.5  # Adjust the gamma value as needed
            look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, look_up_table)
            logging.debug("Applied Histogram Equalization and Gamma Correction for low light.")
        else:
            # Resize image for consistency
            img = cv2.resize(img, (640, 480))
            logging.debug("Resized image for preprocessing.")
        return img
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return img  # Return the original image if preprocessing fails

# -----------------------------
# Email Sending Function
# -----------------------------
def send_email(to_email: str, subject: str, body: str, attachment_path: Optional[str] = None, is_html: bool = False) -> bool:
    """Send an email using SMTP with optional attachment."""
    # Retrieve SMTP configuration from .env
    smtp_server = os.getenv('SMTP_SERVER_GMAIL')
    smtp_port = int(os.getenv('SMTP_PORT_GMAIL', 587))
    sender_email = os.getenv('EMAIL_ADDRESS_GMAIL')
    sender_password = os.getenv('EMAIL_PASSWORD_GMAIL')
    if not sender_email or not sender_password:
        logging.error("Email credentials not set in environment variables.")
        return False  # Indicate failure
    
    try:
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        # Attach the email body
        if is_html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        # Attach the file if provided
        if attachment_path:
            ctype, encoding = mimetypes.guess_type(attachment_path)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            with open(attachment_path, 'rb') as attachment_file:
                attachment = MIMEBase(maintype, subtype)
                attachment.set_payload(attachment_file.read())
                encoders.encode_base64(attachment)
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                msg.attach(attachment)
            logging.debug(f"Attached file {attachment_path} to email.")
        
        # Connect to the SMTP server and send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        logging.info(f"Email sent to {to_email} successfully.")
        return True  # Indicate success
    
    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication Error: Check email credentials or enable App Passwords.")
        return False  # Indicate failure
    except Exception as e:
        logging.error(f"Failed to send email to {to_email}: {e}")
        return False  # Indicate failure

# -----------------------------
# JSON Data Handling Functions
# -----------------------------
def load_or_create_json_file() -> dict:
    """Load data from the JSON file or create one if it doesn't exist."""
    if os.path.exists("recognition_data.json"):
        try:
            with open("recognition_data.json", "r") as json_file:
                data = json.load(json_file)
            logging.info("Recognition data loaded successfully.")
            return data
        except Exception as e:
            logging.error(f"Error loading JSON file: {e}")
            return {
                'dates': [],
                'accuracies': [],
                'confidence_scores': [],
                'match_distances': [],
                'recognition_times': [],
                'lighting_conditions': [],
                'confusion_matrix': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
            }
    else:
        logging.info("Recognition data file not found. Creating a new one.")
        return {
            'dates': [],
            'accuracies': [],
            'confidence_scores': [],
            'match_distances': [],
            'recognition_times': [],
            'lighting_conditions': [],
            'confusion_matrix': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        }

def save_json_data(data: dict):
    """Save recognition data to JSON file."""
    try:
        with open("recognition_data.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
        logging.info("Recognition data saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save recognition data: {e}")

# -----------------------------
# Helper Function to Detect Available Webcam
# -----------------------------
def get_available_camera(max_cameras: int = 5) -> Optional[int]:
    """
    Detects and returns the index of the first available webcam.
    Args:
    max_cameras (int): The maximum number of camera indices to check.
    Returns:
    Optional[int]: The index of the available camera or None if no camera is found.
    """
    for cam_index in range(max_cameras):
        cap = cv2.VideoCapture(cam_index)
        if cap is not None and cap.isOpened():
            cap.release()
            logging.info(f"Webcam found at index {cam_index}.")
            return cam_index
    logging.error("No accessible webcams found.")
    return None
