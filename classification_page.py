import io
import os
import streamlit as st
from PIL import Image
import requests
import pandas as pd
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://127.0.0.1:8000"


def get_suggestions(disease):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)
    prompt = f"""
                You are an agricultural plant disease assistant.
    
                Disease: {disease}
    
                Give a concise, farmer-friendly answer for the Cause, Symptoms, Treatment and Management of the above disease.
    
                Rules:
                - Use simple language.
                - Give practical and safe recommendations.
                - Do not invent pesticide names or dosages.
                - Do not claim this is a definitive diagnosis.
                - Keep the entire response under 180 words.
            """
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            max_output_tokens=200,
        ),
    )
    return response.text


st.set_page_config(page_title="Classification Site", layout="wide")

if "classifier" not in st.session_state:
    st.session_state.classifier = "Plant"

if "prediction_logs" not in st.session_state:
    response = requests.get(f"{API_URL}/predict/logs")

    if response.ok:
        st.session_state.prediction_logs = response.json()
    else:
        st.session_state.prediction_logs = []

st.title("Plant Leaf Disease Classification Site")
st.error("Select the options from the side-bar")


st.sidebar.title("Settings")

if st.sidebar.button("Plant Classifier", width="stretch"):
    st.session_state.classifier = "Plant"

if st.sidebar.button("Plant Disease Remedy Chat", width="stretch"):
    st.session_state.classifier = "Chat"

if st.session_state.classifier == "Plant":
    df = pd.read_csv("./data/plant_disease_dataset.csv")
    class_counts = df["class_name"].value_counts().reset_index()

    class_counts.columns = ["class_name", "count"]

    st.title("Plant Disease Classifier")

    st.bar_chart(
        class_counts,
        x="class_name",
        y="count",
        x_label="Plant Diseases",
        y_label="Number of Samples",
    )

    st.write("Upload a plant leaf image to classify.")

    input_method = st.segmented_control(
        "Choose input method:", ["Upload Image", "Take Picture"], default="Upload Image"
    )

    img = None
    if input_method == "Upload Image":
        img = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
    elif input_method == "Take Picture":
        img = st.camera_input("Take a picture")

    if st.button("Classify", key="classify_plant"):
        time_stamp = time.ctime()
        if img is not None:
            image = Image.open(img).convert("RGB")

            if input_method == "Upload Image":
                st.write("Image Preview")
                st.image(image, width=500)

            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            file_name = (
                getattr(img, "name", "camera_capture.jpg") or "camera_capture.jpg"
            )

            response = requests.post(
                f"{API_URL}/predict/plant",
                files={"file": (file_name, buf.getvalue(), "image/jpeg")},
            )
            result = response.json()

            if result["status"] == "Uncertain":
                st.warning(
                    "High uncertainty detected. This image does not match known classes."
                )
            else:
                st.success(f"Type: {result['prediction']}")

            file_name = (
                getattr(img, "name", "camera_capture.jpg") or "camera_capture.jpg"
            )

            pred_logs = requests.post(
                f"{API_URL}/predict/logs",
                json={
                    "Time Stamp": time_stamp,
                    "Image Name": file_name,
                    "Status": result["status"],
                    "Result": result["prediction"],
                },
            )

            if pred_logs.ok:
                st.session_state.prediction_logs.append(pred_logs.json())
        else:
            st.warning("Please upload an image.")

    if st.session_state.prediction_logs:
        pred_logs_df = pd.DataFrame(st.session_state.prediction_logs)
        st.dataframe(pred_logs_df, width="stretch")

elif st.session_state.classifier == "Chat":
    st.title("Plant Disease Remedy Chat")
    st.info("Put in your Plant Disease Name and get some instant remedies.....")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Enter the disease name from the classifier...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        response = get_suggestions(prompt)
        if response:
            result = response
            st.session_state.messages.append({"role": "assistant", "content": result})

            with st.chat_message("assistant"):
                st.write(result)
        else:
            st.warning("Try again after some time.")
