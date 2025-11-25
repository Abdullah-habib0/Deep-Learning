import gradio as gr
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import traceback
from PIL import Image

print("⚙️ System Initializing (Final Professional Version)...")

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
CNN_PATH = "final_brain_tumor_model.keras"
SEG_PATH = "ResUNet-weights (1).keras"
IMG_SIZE = 128

cnn_model = None
seg_model = None

try:
    if os.path.exists(CNN_PATH):
        cnn_model = tf.keras.models.load_model(CNN_PATH)
        print("✅ CNN Classifier Loaded")
    else:
        print(f"❌ Error: '{CNN_PATH}' not found. Please upload it.")

    if os.path.exists(SEG_PATH):
        seg_model = tf.keras.models.load_model(SEG_PATH, compile=False)
        print("✅ Segmentation Model Loaded")
    else:
        print(f"❌ Error: '{SEG_PATH}' not found. Please upload it.")

except Exception as e:
    print(f"❌ Loading Crash: {e}")

# ==========================================
# 2. XAI FUNCTION (ROBUST REBUILD LOGIC)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Model Graph ko Rebuild karna taake connection error na aye
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = inputs
    layer_output = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer): continue
        try: x = layer(x)
        except: continue
        if layer.name == last_conv_layer_name: layer_output = x

    model_output = x
    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[layer_output, model_output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if preds.shape[-1] == 1: score = preds[0]
        else: score = preds[:, np.argmax(preds[0])]

    grads = tape.gradient(score, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to prevent blank images
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0: return None
    heatmap /= max_val
    return heatmap.numpy()

# ==========================================
# 3. HELPER FUNCTIONS (Drift & Preprocessing)
# ==========================================
def generate_drift_graph(img_array):
    try:
        current_mean = np.mean(img_array)
        mu, sigma = 0.25, 0.1
        x = np.linspace(0, 1, 100)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        plt.figure(figsize=(5, 3))
        plt.plot(x, y, label='Baseline', color='blue')
        plt.fill_between(x, y, alpha=0.2, color='blue')
        plt.axvline(current_mean, color='red', linestyle='--', linewidth=2, label='Current')
        plt.title(f"Drift Check (Mean: {current_mean:.2f})")
        plt.legend(fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = 'drift_status.png'
        plt.savefig(path, dpi=100)
        plt.close()
        return path
    except: return None

def preprocess_cnn(img_pil):
    arr = np.array(img_pil.convert("L").resize((128, 128)), dtype="float32") / 255.0
    return arr.reshape(1, 128, 128, 1)

def preprocess_seg(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img = cv2.resize(img, (256, 256)).astype(np.float64)
    img -= img.mean()
    img /= (img.std() + 1e-8)
    return np.expand_dims(img, axis=0)

# ==========================================
# 4. MAIN LOGIC (WITH HTML CARDS)
# ==========================================
def analyze_brain_mri(image_pil):
    if cnn_model is None: return "❌ System Error: Models not loaded", None, None, None
    if image_pil is None: return "⚠️ Please Upload an Image", None, None, None

    drift_path = generate_drift_graph(np.array(image_pil.convert("L").resize((128,128)))/255.0)

    # Classification Logic
    is_tumor = False
    conf = 0.0
    status_html = ""
    overlay_alpha = 0.4

    try:
        cnn_in = preprocess_cnn(image_pil)
        preds = cnn_model.predict(cnn_in)
        conf = np.max(preds)
        if preds.shape[1] == 1: is_tumor = preds[0][0] > 0.5
        else: is_tumor = np.argmax(preds) == 1

        # --- HTML STATUS LOGIC (Force Colors) ---
        if not is_tumor:
            # HEALTHY (Green Card)
            status_html = f"""
            <div style="background: linear-gradient(to right, #d4edda, #c3e6cb); border-left: 6px solid #28a745; padding: 20px; border-radius: 8px; color: #155724 !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="margin:0; display:flex; align-items:center; color: #155724 !important;">✅ No Tumor Detected</h2>
                <p style="margin-top:5px; font-size: 18px; font-weight:bold; color: #155724 !important;">Confidence: {conf:.2%}</p>
                <small style="color: #155724 !important;">Scan appears normal.</small>
            </div>
            """
        else:
            if 0.50 <= conf < 0.60:
                # WARNING (Yellow Card)
                status_html = f"""
                <div style="background: linear-gradient(to right, #fff3cd, #ffeeba); border-left: 6px solid #ffc107; padding: 20px; border-radius: 8px; color: #856404 !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="margin:0; display:flex; align-items:center; color: #856404 !important;">⚠️ Possible Tumor (Uncertain)</h2>
                    <p style="margin-top:5px; font-size: 18px; font-weight:bold; color: #856404 !important;">Confidence: {conf:.2%}</p>
                    <p style="margin:5px 0 0 0; color: #856404 !important;"><b>Radiologist review is MANDATORY.</b></p>
                </div>
                """
                overlay_alpha = 0.25 # Halka Overlay
            else:
                # TUMOR (Red Card)
                status_html = f"""
                <div style="background: linear-gradient(to right, #f8d7da, #f5c6cb); border-left: 6px solid #dc3545; padding: 20px; border-radius: 8px; color: #721c24 !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="margin:0; display:flex; align-items:center; color: #721c24 !important;">🚨 Tumor Detected</h2>
                    <p style="margin-top:5px; font-size: 18px; font-weight:bold; color: #721c24 !important;">Confidence: {conf:.2%}</p>
                    <small style="color: #721c24 !important;">Immediate clinical correlation recommended.</small>
                </div>
                """
                overlay_alpha = 0.4 # Normal Overlay

    except Exception as e:
        return f"Error: {e}", None, None, drift_path

    # Visuals Logic
    xai_viz = None
    seg_viz = None

    if is_tumor:
        # A. XAI (High Contrast)
        try:
            last_layer_name = next((l.name for l in reversed(cnn_model.layers) if 'conv' in l.name.lower()), None)
            if last_layer_name:
                hm = make_gradcam_heatmap(cnn_in, cnn_model, last_layer_name)
                if hm is not None:
                    hm = np.uint8(255 * hm)
                    heatmap_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                    heatmap_colored = cv2.resize(heatmap_colored, (128, 128))
                    orig = np.array(image_pil.convert("RGB").resize((128,128)))
                    xai_viz = cv2.addWeighted(orig, 0.5, heatmap_colored, 0.5, 0)
        except: pass

        # B. Segmentation (Original Style)
        if seg_model:
            try:
                seg_in = preprocess_seg(image_pil)
                mask = seg_model.predict(seg_in)[0]
                mask = (np.squeeze(mask) > 0.5).astype(np.uint8)

                orig_full = np.array(image_pil.convert("RGB"))
                mask_resized = cv2.resize(mask, (orig_full.shape[1], orig_full.shape[0]))

                overlay = orig_full.copy()
                red_layer = np.zeros_like(orig_full)
                red_layer[:,:,0] = mask_resized * 255

                # Apply Dynamic Alpha
                overlay = cv2.addWeighted(overlay, 1.0, red_layer, overlay_alpha, 0)

                cnts, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, (0,255,0), 2)
                seg_viz = overlay
            except: pass

    return status_html, xai_viz, seg_viz, drift_path

# ==========================================
# 5. GRADIO UI (FINAL VISIBLE HEADINGS)
# ==========================================

# CSS (Headings ko zabardasti dark karne ke liye)
css = """
body {background-color: #f0f2f5;}
.gradio-container {max-width: 1200px !important; margin: auto; padding-top: 20px;}
.header-text {text-align: center; color: #2c3e50; font-family: 'Helvetica', sans-serif;}

/* Disclaimer Box */
.disclaimer-card {
    background-color: #e3f2fd !important;
    border-left: 5px solid #2196f3;
    padding: 15px;
    border-radius: 8px;
    color: #0d47a1 !important;
    font-size: 14px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.disclaimer-card h3, .disclaimer-card li, .disclaimer-card b {
    color: #0d47a1 !important;
}

/* White Background Cards */
.group-box {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Section Titles (Force Dark Color) */
.section-title {
    color: #2c3e50 !important;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}
"""

disclaimer_html = """
<div class="disclaimer-card">
    <h3 style="margin-top:0;">ℹ️ Medical Disclaimer</h3>
    <ul style="padding-left: 20px; margin-bottom:0;">
        <li><b>Screening Only:</b> AI results are preliminary checks, not a final diagnosis.</li>
        <li><b>Imaging Factors:</b> MRI quality & contrast can impact accuracy.</li>
        <li><b>Low Confidence:</b> Results between 50-60% are uncertain.</li>
        <li><b>Consultation:</b> Always verify with a certified Radiologist.</li>
    </ul>
</div>
"""

with gr.Blocks(css, title="NeuroAI Dashboard") as demo:

    gr.HTML("<h1 class='header-text'>🧠 NeuroAI: Advanced Brain Tumor Diagnostics</h1>")

    # --- SECTION 1: INPUT & STATUS ---
    with gr.Row():
        # Left: Input
        with gr.Column(scale=1, elem_classes="group-box"):
            gr.HTML("<h3 class='section-title'>1. Upload Scan</h3>") # Fixed Heading
            inp = gr.Image(label="Input", type="pil", height=250, show_label=False)
            btn = gr.Button("🔍 Run Analysis", variant="primary", size="lg")

        # Right: Disclaimer & Status
        with gr.Column(scale=1):
            gr.HTML(disclaimer_html)
            gr.HTML("<h3 class='section-title'>2. Diagnosis Status</h3>") # Fixed Heading
            status_box = gr.HTML(label="Status")

    # --- SECTION 2: ADVANCED VISUALS ---
    with gr.Row(elem_classes="group-box"):
        with gr.Column():
            gr.HTML("<h3 class='section-title'>Explainability (Why?)</h3>") # Fixed Heading
            xai = gr.Image(label="XAI", show_label=False)
        with gr.Column():
            gr.HTML("<h3 class='section-title'>Localization (Where?)</h3>") # Fixed Heading
            seg = gr.Image(label="Segmentation", show_label=False)
        with gr.Column():
            gr.HTML("<h3 class='section-title'>Data Safety</h3>") # Fixed Heading
            drift = gr.Image(label="Drift", show_label=False)

    # Connect
    btn.click(analyze_brain_mri, inputs=inp, outputs=[status_box, xai, seg, drift])

print("🚀 Launching Perfect UI...")
demo.launch(share=True, debug=True)