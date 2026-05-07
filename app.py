import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Professional Page Config
st.set_page_config(
    page_title="Hessian Surface Inspector", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Metric Cards
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 30px;
    }
</style>
""", unsafe_allow_html=True)

def apply_hessian_analysis(image_np, threshold, blur_k):
    """
    Core Mathematical Analysis Engine.
    """
    # 1. Standardize Resolution for Threshold Consistency
    h, w = image_np.shape[:2]
    target_w = 1000
    target_h = int(h * (target_w / w))
    img_resized = cv2.resize(image_np, (target_w, target_h))
    
    # 2. Preprocessing
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    
    # 3. Hessian Matrix Construction ($Ixx, Iyy, Ixy$)
    # Using float64 to maintain high precision during Det/Trace calculations
    I_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    I_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=3)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=3)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=3)
    
    # Determinant: $Det(H) = (Ixx * Iyy) - (Ixy^2)$
    det_h = (I_xx * I_yy) - (I_xy**2)
    # Trace: $Tr(H) = Ixx + Iyy$
    trace_h = I_xx + I_yy
    
    # 4. Component Masking
    mask = (det_h > threshold).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    
    out_img = img_resized.copy()
    bubbles, pitting = 0, 0
    
    # 5. Iterative Classification
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 12: continue  # Noise filter
        
        # Determine geometric category from Eigenvalue sign (Tr(H))
        avg_trace = np.mean(trace_h[labels == i])
        
        if avg_trace < 0:
            label, color = "BUBBLE", (0, 255, 0)
            bubbles += 1
        else:
            label, color = "PITTING", (0, 0, 255)
            pitting += 1
            
        cv2.rectangle(out_img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(out_img, label, (x, y-8), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
        
    return out_img, bubbles, pitting

# --- APP UI ---
st.title("🛡️ Metal Surface Defect Inspector")
st.caption("Automated Computer Vision Inspection via Hessian Matrix Topology")

# Mathematical Context Expanders
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microscope.png", width=60)
    st.header("Inspection Parameters")
    h_threshold = st.sidebar.slider("Hessiapython -m streamlit run app.pyn Determinant Threshold", 100, 5000, 1000, 
                                     help="Sensitivity of blob detection. Higher = fewer detections.")
    blur_val = st.sidebar.slider("Gaussian Blur Intensity", 1, 15, 5, step=2,
                                help="Reduces noise/texture interference. Must be odd.")
    
    st.markdown("---")
    st.markdown("### Mathematical Theory")
    st.latex(r"H = \begin{bmatrix} I_{xx} & I_{xy} \\ I_{yx} & I_{yy} \end{bmatrix}")
    st.info("""
    **Determinant:** Identifies significant local extremal points.
    **Trace ($Tr < 0$):** Bubble/Protrusion.
    **Trace ($Tr > 0$):** Pitting/Indentation.
    """)

# Main Interface
uploaded_file = st.file_uploader("Upload Image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

if uploaded_file:
    # Read image
    input_image = Image.open(uploaded_file)
    image_np = np.array(input_image)
    
    # Color space fix if necessary (PIL-RGB to BGR for CV2 logic)
    if len(image_np.shape) == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        # Grayscale upload
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    # Core Execution
    with st.spinner("Analyzing Topology..."):
        processed_img, b_count, p_count = apply_hessian_analysis(image_cv, h_threshold, blur_val)
    
    # Dashboard metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Defects", b_count + p_count)
    c2.metric("Bubbles (Light Spot)", b_count)
    c3.metric("Piting (Dark Spot)", p_count)
    
    st.divider()
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Sheet Image")
        st.image(image_np, use_container_width=True)
    
    with col2:
        st.subheader("Automated Detection Result")
        # Convert BGR back to RGB for streamlit display
        res_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)

    # Download Option
    is_success, buffer = cv2.imencode(".png", processed_img)
    if is_success:
        st.download_button(
            label="💾 Download Processed Detection Result",
            data=io.BytesIO(buffer).getvalue(),
            file_name=f"analysis_results_{uploaded_file.name}",
            mime="image/png"
        )
else:
    st.success("Welcome. Please upload a metal sheet photograph to begin the automated inspection.")
    st.markdown("""
    ### Quick Start Guide
    1. **Upload** a clear photo of the metal surface.
    2. Use the **Sidebar Sliders** to adjust for lighting noise or surface texture.
    3. The system will automatically classify and count **Bubbles** and **Pitting** using 2nd-order partial derivatives.
    """)
