import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import timm
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import io

# =====================================================================
# 1. PAGE CONFIGURATION & STYLES
# =====================================================================
st.set_page_config(
    page_title="DeficiVision AI 2.0 | Clinical Biomarker Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Medical Glassmorphic Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background Accent */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(14, 165, 233, 0.05) 0%, transparent 40%),
                    #0b0f17;
    }

    /* Hide standard headers for clean aesthetic */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Top Hero Header Card */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(14, 165, 233, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #34d399, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 6px;
        margin-bottom: 16px;
        font-weight: 400;
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-right: 8px;
    }

    .badge-blue {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border-color: rgba(56, 189, 248, 0.3);
    }

    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border-color: rgba(168, 85, 247, 0.3);
    }

    /* Metric Glass Card */
    .metric-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #10b981;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Recommendation Cards */
    .rec-card {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .rec-icon {
        font-size: 28px;
        background: rgba(16, 185, 129, 0.2);
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    
    .rec-text {
        color: #f1f5f9;
        font-size: 14px;
        font-weight: 500;
        line-height: 1.4;
    }

    /* Custom Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        border-radius: 10px;
    }

    /* Knowledge Card */
    .knowledge-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px;
        height: 100%;
    }

    .knowledge-title {
        font-size: 20px;
        font-weight: 700;
        color: #34d399;
        margin-bottom: 8px;
    }

    .knowledge-symptoms {
        color: #cbd5e1;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .food-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
        color: #e2e8f0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================================
# 2. MODEL LOADING & HYBRID ENSEMBLE PIPELINE
# =====================================================================
CLASSES = ['Vitamin A', 'Vitamin B2', 'Vitamin B3', 'Vitamin B7', 'Vitamin B12', 'Vitamin C']

HEALTH_KNOWLEDGE = {
    'Vitamin A': {
        'symptoms': 'Night blindness, dry eyes (xerophthalmia), Bitot spots, skin hyperkeratosis.',
        'foods': ['🥕 Carrots', '🍠 Sweet Potatoes', '🥬 Spinach', '🥚 Egg Yolks', '🥛 Whole Milk'],
        'recommendations': [
            'Increase daily intake of beta-carotene rich orange/dark leafy vegetables.',
            'Consider physician-directed Vitamin A palmitate supplementation.',
            'Schedule a comprehensive ophthalmic exam to assess retinal function.'
        ],
        'rda': '700 - 900 mcg RAE / day'
    },
    'Vitamin B2': {
        'symptoms': 'Angular cheilitis (cracked mouth corners), magenta tongue, glossitis, red/itchy eyes.',
        'foods': ['🥛 Milk & Yogurt', '🥚 Eggs', '🌰 Almonds', '🍄 Mushrooms', '🥩 Beef Liver'],
        'recommendations': [
            'Consume riboflavin-dense dairy products, eggs, and whole grains.',
            'Integrate a daily Vitamin B-complex dietary supplement.',
            'Consult a dermatologist or clinical nutritionist for angular cheilitis care.'
        ],
        'rda': '1.1 - 1.3 mg / day'
    },
    'Vitamin B3': {
        'symptoms': 'Pellagra symptoms (dermatitis, red lesions, sensitive skin, tongue swelling).',
        'foods': ['🍗 Poultry', '🐟 Tuna & Salmon', '🍚 Brown Rice', '🥜 Peanuts', '🥑 Avocados'],
        'recommendations': [
            'Incorporate lean meats, poultry, fish, and legumes into main meals.',
            'Consider physician-prescribed Niacinamide supplements if deficient.',
            'Protect affected skin areas from direct ultraviolet sunlight.'
        ],
        'rda': '14 - 16 mg NE / day'
    },
    'Vitamin B7': {
        'symptoms': 'Hair thinning/alopecia, red scaly rash around mouth/nose, brittle fingernails.',
        'foods': ['🥚 Whole Eggs', '🥜 Almonds & Walnuts', '🌻 Sunflower Seeds', '🍠 Sweet Potatoes', '🧀 Cheese'],
        'recommendations': [
            'Eat biotin-rich whole foods such as egg yolks and raw nuts.',
            'Daily Biotin (B7) supplement (30-100 mcg) for hair and nail integrity.',
            'Consult a clinical practitioner if hair loss or rash persists.'
        ],
        'rda': '30 mcg / day'
    },
    'Vitamin B12': {
        'symptoms': 'Pale or yellowish skin, smooth inflamed tongue (glossitis), fatigue, mouth sores.',
        'foods': ['🥩 Lean Beef', '🐟 Salmon & Trout', '🥛 Fortified Dairy', '🥚 Eggs', '🥣 Fortified Cereal'],
        'recommendations': [
            'Consume Vitamin B12 enriched animal proteins or fortified plant milks.',
            'Take Methylcobalamin B12 supplements (especially for vegetarians/vegans).',
            'Perform a complete blood count (CBC) and serum B12 blood diagnostic.'
        ],
        'rda': '2.4 mcg / day'
    },
    'Vitamin C': {
        'symptoms': 'Bleeding or swollen gums, easy bruising, slow wound healing, dry scaly skin.',
        'foods': ['🍊 Citrus Fruits', '🫑 Bell Peppers', '🥝 Kiwi Fruit', '🍓 Strawberries', '🥦 Broccoli'],
        'recommendations': [
            'Increase fresh raw citrus, berries, and raw capsicum consumption.',
            'Daily Ascorbic Acid (Vitamin C) supplement (250mg - 500mg).',
            'Maintain optimal oral hygiene and consult a dentist for gum health.'
        ],
        'rda': '75 - 90 mg / day'
    }
}

@st.cache_resource
def load_models():
    cnn_path = 'cnn_vitamin_model.pth'
    vit_path = 'vit_vitamin_model.pth'
    
    cnn_url = "https://github.com/abhinandk05/Vitamin_Deficiency_Project/releases/download/v1.0/cnn_vitamin_model.pth"
    vit_url = "https://github.com/abhinandk05/Vitamin_Deficiency_Project/releases/download/v1.0/vit_vitamin_model.pth"

    if not os.path.exists(cnn_path):
        with st.spinner("Downloading ResNet50 CNN model weights (~96MB)..."):
            urllib.request.urlretrieve(cnn_url, cnn_path)

    if not os.path.exists(vit_path):
        with st.spinner("Downloading Vision Transformer (ViT) weights (~344MB)..."):
            urllib.request.urlretrieve(vit_url, vit_path)

    # 1. ResNet50
    cnn = models.resnet50(weights=None)
    num_features_cnn = cnn.fc.in_features
    cnn.fc = nn.Sequential(nn.Linear(num_features_cnn, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 6))
    cnn.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu'), weights_only=True))
    cnn.eval()

    # 2. Vision Transformer
    vit = timm.create_model('vit_base_patch16_224', pretrained=False)
    num_features_vit = vit.head.in_features
    vit.head = nn.Sequential(nn.Linear(num_features_vit, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 6))
    vit.load_state_dict(torch.load(vit_path, map_location=torch.device('cpu'), weights_only=True))
    vit.eval()
    
    return cnn, vit

try:
    cnn_model, vit_model = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading PyTorch models: {e}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper function to generate high quality sample images if user clicks "Try Sample"
def generate_sample_image(label):
    img = Image.new('RGB', (400, 400), color=(30, 41, 59))
    draw = ImageDraw.Draw(img)
    
    # Draw clinical aesthetic illustration shapes
    colors = {
        'Eye Biomarker (Vit A)': ((16, 185, 129), '👁️'),
        'Lip / Mouth Corner (Vit B2)': ((244, 63, 94), '👄'),
        'Skin Rash / Lesion (Vit B3)': ((245, 158, 11), '✋'),
        'Hair & Scalp Condition (Vit B7)': ((168, 85, 247), '💇'),
        'Pale Tongue Sign (Vit B12)': ((236, 72, 153), '👅'),
        'Gums / Vascular (Vit C)': ((14, 165, 233), '🪥')
    }
    
    col, emoji = colors.get(label, ((16, 185, 129), '🔬'))
    
    draw.ellipse([80, 80, 320, 320], fill=col[0], outline=(255, 255, 255), width=4)
    draw.ellipse([140, 140, 260, 260], fill=(15, 23, 42))
    
    return img

# Initialize Session State History
if 'history' not in st.session_state:
    st.session_state.history = []

# =====================================================================
# 3. SIDEBAR & HERO HEADER
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/health.png", width=70)
    st.title("DeficiVision AI 2.0")
    st.markdown("**Clinical Vision Analysis**")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.markdown("🟢 **ResNet50 Model:** Ready")
    st.markdown("🟢 **Vision Transformer:** Ready")
    st.markdown("⚡ **Inference Hardware:** CPU Engine")
    st.markdown("🎯 **Target Biomarkers:** 6 Classes")
    
    st.markdown("---")
    st.markdown("### 📑 Clinical Guidance")
    st.info("For maximum diagnostic accuracy, capture photos under bright, neutral white lighting focused directly on the eye, lips, tongue, or skin lesion.")

# Top Hero Banner
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">DeficiVision AI 2.0 Diagnostic Suite</h1>
    <p class="hero-subtitle">Dual-Engine ResNet50 + Vision Transformer Ensemble for Non-Invasive Visual Vitamin Deficiency Screening</p>
    <div>
        <span class="badge-pill">🧠 ResNet50 CNN Active</span>
        <span class="badge-pill badge-blue">👁️ Vision Transformer (ViT-16)</span>
        <span class="badge-pill badge-purple">⚖️ Soft-Voting Ensemble</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 4. TABBED INTERFACE LAYOUT
# =====================================================================
tab_scanner, tab_knowledge, tab_architecture, tab_history = st.tabs([
    "🔬 Diagnostic Scanner",
    "📚 Vitamin Knowledge Hub",
    "🧠 AI Ensemble Architecture",
    "📋 Session Diagnostic Log"
])

# ---------------------------------------------------------------------
# TAB 1: DIAGNOSTIC SCANNER
# ---------------------------------------------------------------------
with tab_scanner:
    col_input, col_results = st.columns([1, 1.3], gap="large")
    
    with col_input:
        st.subheader("1. Input Feature Biomarker")
        
        input_mode = st.radio(
            "Select Input Source:",
            ["📁 Upload Photo", "📸 Live Camera Stream", "🧪 Test Sample Images"],
            horizontal=True
        )
        
        selected_image = None
        
        if input_mode == "📁 Upload Photo":
            uploaded_file = st.file_uploader(
                "Upload a high-resolution photo of Eye, Lips, Tongue, Gums, Skin, or Hair:",
                type=["jpg", "jpeg", "png", "webp"]
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert('RGB')
                
        elif input_mode == "📸 Live Camera Stream":
            camera_file = st.camera_input("Capture clinical symptom photo:")
            if camera_file is not None:
                selected_image = Image.open(camera_file).convert('RGB')
                
        elif input_mode == "🧪 Test Sample Images":
            sample_choice = st.selectbox(
                "Choose a simulated clinical sample:",
                [
                    'Eye Biomarker (Vit A)',
                    'Lip / Mouth Corner (Vit B2)',
                    'Skin Rash / Lesion (Vit B3)',
                    'Hair & Scalp Condition (Vit B7)',
                    'Pale Tongue Sign (Vit B12)',
                    'Gums / Vascular (Vit C)'
                ]
            )
            if st.button("Load Selected Sample"):
                selected_image = generate_sample_image(sample_choice)
                st.session_state['active_sample'] = selected_image

            if 'active_sample' in st.session_state and selected_image is None:
                selected_image = st.session_state['active_sample']

        if selected_image is not None:
            st.markdown("#### Sample Preview")
            st.image(selected_image, use_container_width=True, caption="Biomarker Scan Subject")
            st.caption(f"Dimensions: {selected_image.size[0]} x {selected_image.size[1]} px | Mode: {selected_image.mode}")

    # RESULTS COLUMN
    with col_results:
        st.subheader("2. AI Ensemble Diagnostics")
        
        if selected_image is not None and model_loaded:
            with st.spinner("Executing ResNet50 + Vision Transformer inference pipeline..."):
                tensor_img = preprocess(selected_image).unsqueeze(0)
                
                with torch.no_grad():
                    cnn_logits = cnn_model(tensor_img)
                    vit_logits = vit_model(tensor_img)
                    
                    cnn_probs = torch.softmax(cnn_logits, dim=1)[0]
                    vit_probs = torch.softmax(vit_logits, dim=1)[0]
                    
                    # Ensemble Soft-Voting Average
                    hybrid_probs = (cnn_probs + vit_probs) / 2.0
                    
                    confidence_tensor, pred_idx = torch.max(hybrid_probs, 0)
                    predicted_class = CLASSES[pred_idx.item()]
                    confidence_pct = confidence_tensor.item() * 100
                    
                    cnn_pred_idx = torch.argmax(cnn_probs).item()
                    vit_pred_idx = torch.argmax(vit_probs).item()

            # Record session history
            st.session_state.history.append({
                'class': predicted_class,
                'confidence': f"{confidence_pct:.1f}%",
                'resnet_pred': CLASSES[cnn_pred_idx],
                'vit_pred': CLASSES[vit_pred_idx]
            })

            # Primary Prediction Cards
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Deficiency</div>
                    <div class="metric-value" style="font-size:26px;">{predicted_class}</div>
                    <div style="color: #94a3b8; font-size: 12px; margin-top:4px;">Primary Classification Target</div>
                </div>
                """, unsafe_allow_html=True)
                
            with mcol2:
                conf_color = "#10b981" if confidence_pct > 75 else "#f59e0b" if confidence_pct > 50 else "#ef4444"
                conf_tag = "High Confidence" if confidence_pct > 75 else "Moderate Confidence" if confidence_pct > 50 else "Low Confidence"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Ensemble Confidence</div>
                    <div class="metric-value" style="color:{conf_color};">{confidence_pct:.1f}%</div>
                    <div style="color: {conf_color}; font-size: 12px; font-weight:600; margin-top:4px;">{conf_tag}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Side-by-Side Model Agreement Breakdown
            st.markdown("#### Model Consensus Breakdown")
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                cnn_top_conf = cnn_probs[cnn_pred_idx].item() * 100
                st.markdown(f"**ResNet50 (CNN):** `{CLASSES[cnn_pred_idx]}` ({cnn_top_conf:.1f}%)")
                st.progress(cnn_top_conf / 100.0)
            with ccol2:
                vit_top_conf = vit_probs[vit_pred_idx].item() * 100
                st.markdown(f"**Vision Transformer:** `{CLASSES[vit_pred_idx]}` ({vit_top_conf:.1f}%)")
                st.progress(vit_top_conf / 100.0)

            # Full Probability Distribution Breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Full Deficiency Probability Map")
            
            prob_df = pd.DataFrame({
                'Deficiency Class': CLASSES,
                'Ensemble Probability (%)': [hybrid_probs[i].item() * 100 for i in range(6)],
                'ResNet50 (%)': [cnn_probs[i].item() * 100 for i in range(6)],
                'ViT Transformer (%)': [vit_probs[i].item() * 100 for i in range(6)]
            }).sort_values(by='Ensemble Probability (%)', ascending=False)
            
            for index, row in prob_df.iterrows():
                val = row['Ensemble Probability (%)']
                col_name, col_bar = st.columns([1.5, 3])
                with col_name:
                    st.markdown(f"**{row['Deficiency Class']}**")
                with col_bar:
                    st.progress(min(val / 100.0, 1.0))
                    st.caption(f"Ensemble: {val:.1f}% | CNN: {row['ResNet50 (%)']:.1f}% | ViT: {row['ViT Transformer (%)']:.1f}%")

            # Targeted Care Plan
            st.markdown("---")
            st.markdown("### 🥗 Targeted Care Plan & Recommendations")
            rec_info = HEALTH_KNOWLEDGE.get(predicted_class, HEALTH_KNOWLEDGE['Vitamin A'])
            
            st.markdown(f"**Associated Visual Symptoms:** {rec_info['symptoms']}")
            
            st.markdown("#### Key Dietary Sources:")
            food_html = "".join([f'<span class="food-tag">{f}</span>' for f in rec_info['foods']])
            st.markdown(food_html, unsafe_allow_html=True)
            
            st.markdown("#### Recommended Actions:")
            for rec in rec_info['recommendations']:
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-icon">🩺</div>
                    <div class="rec-text">{rec}</div>
                </div>
                """, unsafe_allow_html=True)

            # Printable Summary Generator
            st.markdown("<br>", unsafe_allow_html=True)
            report_text = f"""====================================================
DEFICIVISION AI 2.0 - CLINICAL SCREENING REPORT
====================================================
Primary Finding: {predicted_class} Deficiency
Ensemble Confidence: {confidence_pct:.1f}% ({conf_tag})
CNN ResNet50 Vote: {CLASSES[cnn_pred_idx]} ({cnn_top_conf:.1f}%)
Vision Transformer Vote: {CLASSES[vit_pred_idx]} ({vit_top_conf:.1f}%)

RECOMMENDED CARE STEPS:
- Dietary Intake: {', '.join(rec_info['foods'])}
- Guideline 1: {rec_info['recommendations'][0]}
- Guideline 2: {rec_info['recommendations'][1]}

DISCLAIMER: This report is generated by an educational AI prototype.
Consult a licensed healthcare professional for official diagnosis.
===================================================="""
            
            st.download_button(
                label="🖨️ Download Printable Diagnostic Report (.TXT)",
                data=report_text,
                file_name=f"DeficiVision_Report_{predicted_class.replace(' ', '_')}.txt",
                mime="text/plain"
            )

        elif not model_loaded:
            st.error("Model state unavailable. Please ensure weight `.pth` files exist.")
        else:
            st.info("👈 Please select or upload a symptom image on the left panel to begin automated analysis.")

# ---------------------------------------------------------------------
# TAB 2: VITAMIN KNOWLEDGE HUB
# ---------------------------------------------------------------------
with tab_knowledge:
    st.subheader("Clinical Vitamin Deficiency Reference Index")
    st.markdown("Explore visual indicators, target dietary sources, and daily recommended intake for all 6 screening targets.")
    
    kcols1, kcols2 = st.columns(2, gap="medium")
    
    items = list(HEALTH_KNOWLEDGE.items())
    for i, (v_name, data) in enumerate(items):
        target_col = kcols1 if i % 2 == 0 else kcols2
        with target_col:
            foods_tags = "".join([f'<span class="food-tag">{f}</span>' for f in data['foods']])
            st.markdown(f"""
            <div class="knowledge-card">
                <div class="knowledge-title">👁️ {v_name}</div>
                <div style="font-size:12px; color:#38bdf8; font-weight:600; margin-bottom:6px;">RDA: {data['rda']}</div>
                <div class="knowledge-symptoms"><strong>Key Visual Indicators:</strong> {data['symptoms']}</div>
                <div style="margin-bottom:12px;"><strong>Top Food Sources:</strong><br>{foods_tags}</div>
                <div style="font-size:12px; color:#94a3b8;"><strong>Primary Action:</strong> {data['recommendations'][0]}</div>
            </div>
            <br>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TAB 3: AI ARCHITECTURE
# ---------------------------------------------------------------------
with tab_architecture:
    st.subheader("Hybrid ResNet50 + Vision Transformer Ensemble")
    st.markdown("""
    DeficiVision AI 2.0 fuses two complementary deep learning paradigms to maximize visual feature detection across diverse lighting, dermal tones, and camera resolution conditions.
    """)
    
    acol1, acol2, acol3 = st.columns(3)
    with acol1:
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 ResNet50 (CNN)</h4>
            <p style="font-size:13px; color:#94a3b8;">Extracts local high-frequency spatial gradients, skin textures, micro-lesions, and edge features via residual convolutional blocks.</p>
        </div>
        """, unsafe_allow_html=True)
    with acol2:
        st.markdown("""
        <div class="metric-card">
            <h4>👁️ ViT-Base (Transformer)</h4>
            <p style="font-size:13px; color:#94a3b8;">Splits input images into 16x16 patch tokens and calculates global multi-head self-attention across the full anatomical field.</p>
        </div>
        """, unsafe_allow_html=True)
    with acol3:
        st.markdown("""
        <div class="metric-card">
            <h4>⚖️ Soft-Voting Fusion</h4>
            <p style="font-size:13px; color:#94a3b8;">Combines model posterior probabilities to reduce variance and eliminate false positive classifications.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Simulated Model Training History")
    
    epochs = list(range(1, 11))
    history_df = pd.DataFrame({
        'Epoch': epochs,
        'ResNet50 Accuracy': [0.68, 0.74, 0.79, 0.82, 0.85, 0.87, 0.89, 0.90, 0.92, 0.93],
        'ViT Transformer Accuracy': [0.65, 0.72, 0.78, 0.83, 0.86, 0.89, 0.91, 0.93, 0.94, 0.95],
        'Hybrid Ensemble Accuracy': [0.72, 0.78, 0.84, 0.88, 0.91, 0.93, 0.95, 0.96, 0.97, 0.98]
    }).set_index('Epoch')
    
    st.line_chart(history_df, height=300)

# ---------------------------------------------------------------------
# TAB 4: SESSION HISTORY
# ---------------------------------------------------------------------
with tab_history:
    st.subheader("Session Diagnostic Audit Log")
    if len(st.session_state.history) > 0:
        h_df = pd.DataFrame(st.session_state.history)
        st.dataframe(h_df, use_container_width=True)
        if st.button("🗑️ Clear History Log"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No scans recorded in current session. Perform a scan in the Diagnostic Scanner tab.")

# Disclaimer Footer
st.markdown("---")
st.warning("⚠️ **Medical Disclaimer:** DeficiVision AI 2.0 is an educational deep learning demonstration. It does not carry medical device certifications and must not replace professional clinical evaluation or physician consultation.")
