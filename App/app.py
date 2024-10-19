import sys
import os

current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(current_path)

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import psycopg2
import pandas as pd

from utils.utils import *  # Your utility functions

# Load the YOLO models
detector = YOLO(current_path + '/Model/LP_Detect_YOLOv11n.pt')  # License Plate Detection model
recognizer = YOLO(current_path + '/Model/LP_Recog_YOLOv11n.pt')  # License Plate Character Recognition model

# Database connection function
def connect_to_database(database_name, database_host, database_user, database_password, database_port):
    try:
        conn = psycopg2.connect(database=database_name,
                                host=database_host,
                                user=database_user,
                                password=database_password,
                                port=database_port)
        st.success("Connected to database.")
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Function for taking camera input
def get_camera_input():
    cap = cv2.VideoCapture("https://192.168.100.101:8080/video")
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading from camera.")
            break

        frame, plate = process_frame(frame, detector, recognizer)
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        st.session_state['frame'] = frame

        if st.session_state['take_snapshot']:
            cap.release()
            st.session_state['take_snapshot'] = False
            return plate

    cap.release()

# Function for taking video input
def get_video_input(video_source):
    skip_frames = 5

    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(video_source)

    cap = st.session_state.cap
    frame_placeholder = st.empty()

    while cap.isOpened():
        for _ in range(skip_frames):
            ret, frame = cap.read()
            if not ret:
                st.warning("No frames available.")
                break

        processed_frame, plate = process_frame(frame, detector, recognizer)
        frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
        st.session_state['frame'] = frame

        if st.session_state['take_snapshot']:
            st.session_state['take_snapshot'] = False
            return frame, plate

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    st.success("Video ended.")

# Function for verifying license plate in database
def verify_license_plate(plate, conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_name, msv FROM parking_users WHERE plate = %s", (plate,))
        result = cur.fetchone()
        return result
    except Exception as e:
        st.error(f"Error during verification: {e}")
        return None

# Function to add license plate to database
def add_to_database(plate, owner, msv, conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(id_user), 0) + 1 FROM parking_users")
        new_id = cur.fetchone()[0]
        cur.execute("INSERT INTO parking_users (id_user, plate, user_name, msv) VALUES (%s, %s, %s, %s)",
                    (new_id, plate, owner, msv))
        conn.commit()
        st.success("License plate added to database.")
    except Exception as e:
        st.error(f"Error adding to database: {e}")

# Main function
def main():
    # Initialize session state variables
    if 'plate' not in st.session_state:
        st.session_state['plate'] = ''
    if 'form_submitted' not in st.session_state:
        st.session_state['form_submitted'] = False
    if 'take_snapshot' not in st.session_state:
        st.session_state['take_snapshot'] = False
    if 'name' not in st.session_state:
        st.session_state['name'] = ''
    if 'msv' not in st.session_state:
        st.session_state['msv'] = 0

    video_path = current_path + "/App/test_vid.MOV"
    st.title("License Plate Detection and Verification")

    # Database connection
    with st.spinner("Loading data from database..."):
        database_name = st.text_input("Enter database name:")
        database_host = st.text_input("Enter database host:")
        database_user = st.text_input("Enter database user:")
        database_password = st.text_input("Enter database password:")
        database_port = st.text_input("Enter database port:")
        conn = connect_to_database(database_name, database_host, database_user, database_password, database_port)

    # Ensure the connection is valid
    if conn is None:
        return  # Exit the app if no database connection

    # Video source selection
    selection = st.radio("Select video source:", ("Camera", "Video File"))
    
    # Handle video input
    if selection == "Camera":
        col1, col2 = st.columns(2)
        with col2:
            snapshot_button = st.button("Take Snapshot")
            if snapshot_button and st.session_state['plate'] == '':
                st.session_state['take_snapshot'] = True  # Set flag to True
            continue_button = st.button("Continue")
            if continue_button:
                st.session_state['plate'] = ''
                pass

        with col1:
            if st.session_state['plate'] == '':
                plate = get_camera_input()
                if plate:
                    st.session_state['plate'] = plate  # Store detected plate in session state
            else:
                st.image(st.session_state['frame'], channels="BGR", use_column_width=True)

    elif selection == "Video File":
        col1, col2 = st.columns(2)

        with col2:
            snapshot_button = st.button("Take Snapshot")
            if snapshot_button and st.session_state['plate'] == '':
                st.session_state['take_snapshot'] = True
            continue_button = st.button("Continue")
            if continue_button:
                st.session_state['plate'] = ''
                pass

        with col1:
            if st.session_state['plate'] == '':
                frame, plate = get_video_input(video_path)
                if plate:
                    st.session_state['plate'] = plate  # Store detected plate in session state
            else:
                st.image(st.session_state['frame'], channels="BGR", use_column_width=True)
                

    # License plate handling and form display
    with col2:
        if st.session_state['plate'] and not st.session_state['form_submitted']:
            plate = st.session_state['plate']
            st.write("Detected license Plate: ", plate)

            owner = verify_license_plate(plate, conn)
            if owner:
                st.success(f"Verified! Plate owner: {owner[0]}, MSV: {owner[1]}")
            else:
                st.error("Not verified. Add to database:")

                with st.form("add_to_database"):
                    st.session_state['name'] = st.text_input("Enter owner's name:")
                    st.session_state['msv'] = st.number_input("Enter owner's MSV number:", min_value=0, max_value=99999999)
                    st.session_state['form_submitted'] = st.form_submit_button("Submit")
        else:
            st.warning("No license plate detected.")
    
    if st.session_state['form_submitted'] and st.session_state['plate']:
        plate = st.session_state['plate']
        name = st.session_state['name']
        msv = st.session_state['msv']
        add_to_database(plate, name, msv, conn)

        st.success("License plate added to database. Ready for new plate detection.")
        st.session_state['plate'] = ''  # Reset plate to allow new detection
        st.session_state['form_submitted'] = False  # Reset form submission flag
        
        # print the database
        cur = conn.cursor()
        cur.execute("SELECT * FROM parking_users")
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
        st.write(df)

if __name__ == "__main__":
    main()
