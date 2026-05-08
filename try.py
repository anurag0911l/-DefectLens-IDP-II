import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hessian Surface Inspector", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; }
    .stAlert { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

def apply_hessian_analysis(image_np, threshold, blur_k, min_area):
    """
    Analyzes surface topology using the Hessian Matrix.
    Filters out small noise based on the min_area parameter.
    """
    # 1. Standardize Resolution for consistent math across different uploads
    h, w = image_np.shape[:2]
    target_w = 1000
    target_h = int(h * (target_w / w))
    img_resized = cv2.resize(image_np, (target_w, target_h))
    
    # 2. Preprocessing
    # Increasing Blur (blur_k) is the best way to ignore tiny 'grainy' texture
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    
    # 3. Hessian Matrix Construction ($Ixx, Iyy, Ixy$)
    I_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    I_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=3)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=3)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=3)
    
    # Determinant identifies the 'blobs'
    det_h = (I_xx * I_yy) - (I_xy**2)
    # Trace identifies if it's a peak (bubble) or valley (pit)
    trace_h = I_xx + I_yy
    
    # 4. Component Masking
    mask = (det_h > threshold).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    
    out_img = img_resized.copy()
    bubbles, pitting = 0, 0
    
    # 5. Iterative Classification
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # FOCUS CONTROL: Ignore defects smaller than the user-defined area
        if area < min_area: 
            continue
        
        # Calculate the average trace within the detected blob
        avg_trace = np.mean(trace_h[labels == i])
        
        if avg_trace < 0:
            label, color = "BUBBLE", (0, 255, 0) # Green for protrusions
            bubbles += 1
        else:
            label, color = "PITTING", (0, 0, 255) # Red for indentations
            pitting += 1
            
        cv2.rectangle(out_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(out_img, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    return out_img, bubbles, pitting

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microscope.png", width=60)
    st.header("Inspection Tuning")
    
    h_threshold = st.slider("Hessian Sensitivity", 100, 10000, 2000, 
                            help="Higher values ignore subtle surface texture.")
    
    blur_val = st.slider("Surface Smoothing (Blur)", 1, 25, 7, step=2,
                         help="Increase this to 'melt away' tiny dots and focus on large defects.")
    
    min_area = st.slider("Minimum Defect Size (Pixels)", 5, 2000, 100,
                         help="Any detection smaller than this area will be ignored.")
    
    st.divider()
    st.markdown("### Topology Logic")
    st.info("The algorithm uses 2nd-order derivatives to find local maxima/minima on the metal surface.")

# --- MAIN INTERFACE ---
st.title("🛡️ Metal Surface Defect Inspector")
st.caption("Professional Grade Topology Analysis for Industrial Quality Control")

uploaded_file = st.file_uploader("Upload Surface Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # Load and prepare image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Analyzing Surface Topology..."):
        processed_img, b_count, p_count = apply_hessian_analysis(image_cv, h_threshold, blur_val, min_area)
    
    # Metrics display
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Detected", b_count + p_count)
    m2.metric("Bubbles (Light)", b_count)
    m3.metric("Pitting (Dark)", p_count)
    
    st.divider()
    
    # Result comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("Detected Defects")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Download Button
    _, buffer = cv2.imencode(".png", processed_img)
    st.download_button(
        label="📥 Download Result Image",
        data=io.BytesIO(buffer).getvalue(),
        file_name=f"inspected_{uploaded_file.name}",
        mime="image/png"
    )
else:
    st.success("Ready for analysis. Please upload an image to begin.")
