import streamlit as st
from observer import load_model, detect_vehicles
from PIL import Image
from io import BytesIO

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🚗",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Detection System")
st.markdown(
    "Upload a road or traffic image to detect and count vehicles. "
    "The model identifies **cars**, **vans**, and **buses**."
)
st.divider()

# ── Model Loading (cached — only loads once per session) ──────────────────────
@st.cache_resource
def get_model():
    return load_model("best_vehicle2.pt")

model = get_model()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_pil   = Image.open(BytesIO(image_bytes))

    st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    st.divider()

    # ── Detect Button ─────────────────────────────────────────────────────────
    if st.button("🔍 Detect Vehicles"):
        with st.spinner("Running detection..."):
            annotated_image, class_counts = detect_vehicles(image_pil, model)

        # ── Result Image ──────────────────────────────────────────────────────
        st.subheader("Detection Result")
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)
        st.divider()

        # ── Count Summary ─────────────────────────────────────────────────────
        st.subheader("Vehicle Count Summary")

        if class_counts:
            cols = st.columns(len(class_counts) + 1)

            # One metric card per class
            for i, (class_name, count) in enumerate(sorted(class_counts.items())):
                cols[i].metric(label=class_name.capitalize(), value=count)

            # Total count in the last column
            cols[-1].metric(label="Total", value=sum(class_counts.values()))

        else:
            st.warning("No vehicles detected in this image.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: YOLOv12n · Classes: car, van, bus · Capstone Project Module 4 · Purwadhika AI Engineer Bootcamp")