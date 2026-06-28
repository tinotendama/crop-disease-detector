import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Detector — Malta Academy",
    page_icon="🌿",
    layout="wide"
)

# ── 38 PlantVillage class names (index must match your model's training order) ─
CLASS_NAMES = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Powdery Mildew",
    "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot (Gray Leaf Spot)",
    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Corn - Healthy",
    "Grape - Black Rot",
    "Grape - Black Measles (Esca)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Pepper - Bacterial Spot",
    "Pepper - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two-spotted Spider Mite)",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy",
]

# ── Treatment recommendations ─────────────────────────────────────────────────
TREATMENTS = {
    "Apple - Apple Scab":
        "Apply fungicide (captan or myclobutanil) at bud break. Rake and destroy fallen leaves. Plant resistant varieties where possible.",
    "Apple - Black Rot":
        "Prune and destroy infected branches. Apply copper-based fungicide. Remove mummified fruit from the tree.",
    "Apple - Cedar Apple Rust":
        "Apply protective fungicide during spring. Remove nearby juniper/cedar trees if possible. Use resistant apple varieties.",
    "Cherry - Powdery Mildew":
        "Apply sulfur or potassium bicarbonate fungicide. Improve air circulation by pruning. Avoid excess nitrogen fertiliser.",
    "Corn - Cercospora Leaf Spot (Gray Leaf Spot)":
        "Apply triazole or strobilurin fungicide at tasselling. Plant resistant hybrid varieties. Rotate crops annually.",
    "Corn - Common Rust":
        "Apply fungicide early at first sign. Plant rust-resistant hybrids. Monitor fields regularly during warm humid weather.",
    "Corn - Northern Leaf Blight":
        "Apply fungicide at tasselling if disease is present. Use resistant varieties. Practice crop rotation.",
    "Grape - Black Rot":
        "Apply mancozeb or myclobutanil fungicide from bud break. Remove all mummified fruit. Ensure good canopy airflow.",
    "Grape - Black Measles (Esca)":
        "No curative treatment available. Remove and destroy infected wood. Apply preventive wound sealant after pruning.",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)":
        "Apply copper-based fungicide. Improve vineyard air circulation. Remove infected leaves promptly.",
    "Orange - Haunglongbing (Citrus Greening)":
        "No cure exists. Remove and destroy infected trees immediately to prevent spread. Control the Asian citrus psyllid vector with insecticide.",
    "Peach - Bacterial Spot":
        "Apply copper bactericide from early spring. Avoid overhead irrigation. Plant resistant varieties.",
    "Pepper - Bacterial Spot":
        "Use copper-based bactericide. Avoid working with plants when wet. Use certified disease-free seed.",
    "Potato - Early Blight":
        "Apply chlorothalonil or mancozeb fungicide. Ensure adequate plant nutrition — stressed plants are most susceptible. Rotate crops.",
    "Potato - Late Blight":
        "Apply metalaxyl or cymoxanil fungicide immediately. Destroy all infected plant material. Do not leave infected tubers in the soil.",
    "Squash - Powdery Mildew":
        "Apply potassium bicarbonate or sulfur fungicide. Improve airflow around plants. Avoid overhead watering.",
    "Strawberry - Leaf Scorch":
        "Remove infected leaves. Apply fungicide (captan or myclobutanil). Ensure good drainage and air circulation.",
    "Tomato - Bacterial Spot":
        "Apply copper bactericide. Remove infected plant debris. Avoid overhead watering and working with wet plants.",
    "Tomato - Early Blight":
        "Apply chlorothalonil fungicide at first sign. Mulch around plants to prevent soil splash. Remove lower infected leaves.",
    "Tomato - Late Blight":
        "Apply copper-based fungicide immediately. Remove and destroy all infected material. Avoid overhead watering.",
    "Tomato - Leaf Mold":
        "Improve greenhouse ventilation. Apply mancozeb or chlorothalonil fungicide. Reduce humidity below 85%.",
    "Tomato - Septoria Leaf Spot":
        "Apply chlorothalonil fungicide. Remove infected leaves. Rotate crops and avoid overhead watering.",
    "Tomato - Spider Mites (Two-spotted Spider Mite)":
        "Apply miticide (abamectin or spiromesifen). Spray undersides of leaves thoroughly. Increase humidity — mites thrive in dry conditions.",
    "Tomato - Target Spot":
        "Apply chlorothalonil or mancozeb fungicide. Remove infected debris. Improve air circulation.",
    "Tomato - Yellow Leaf Curl Virus":
        "No cure — remove and destroy infected plants immediately. Control whitefly vectors with insecticide. Use resistant varieties.",
    "Tomato - Mosaic Virus":
        "No cure — remove and destroy infected plants. Wash hands and tools thoroughly. Control aphid vectors.",
}

HEALTHY_CLASSES = [c for c in CLASS_NAMES if "Healthy" in c]
# ── Treatment recommendations — Shona (sn) ────────────────────────────────────
TREATMENTS_SN = {
    "Apple - Apple Scab":
        "Shandisa mushonga we fungicide (captan kana myclobutanil) panguva yekutanga kumera. Unganidza uye uparadze mashizha akadonha. Sima mhando dzine kudzivirira.",
    "Apple - Black Rot":
        "Cheka uye uparadze matavi akabatwa. Shandisa fungicide ine copper. Bvisa michero yakaomera pamuti.",
    "Apple - Cedar Apple Rust":
        "Shandisa fungicide yekudzivirira muchirimo. Bvisa miti ye cedar/juniper iri pedyo kana zvichibvira. Shandisa mhando dzine kudzivirira.",
    "Cherry - Powdery Mildew":
        "Shandisa sulfur kana potassium bicarbonate fungicide. Vandudza mhepo nekucheka. Dzivisa fertilizer ine nitrogen yakawandisa.",
    "Corn - Cercospora Leaf Spot (Gray Leaf Spot)":
        "Shandisa fungicide panguva yekutanga kubereka. Sima mhando dzine kudzivirira. Chinja zvirimwa gore negore.",
    "Corn - Common Rust":
        "Shandisa fungicide nokukasika pakuona kwekutanga. Sima mhando dzine kudzivirira. Tarisa minda nguva dzose mukati menguva ine kupisa nemvura.",
    "Corn - Northern Leaf Blight":
        "Shandisa fungicide panguva yekubereka kana chirwere chichionekwa. Shandisa mhando dzine kudzivirira. Ita kuchinja zvirimwa.",
    "Grape - Black Rot":
        "Shandisa mancozeb kana myclobutanil fungicide kubva pakutanga kumera. Bvisa michero yose yakaomera. Chengetedza mhepo yakanaka.",
    "Grape - Black Measles (Esca)":
        "Hapana mushonga unorapa. Bvisa uye uparadze huni dzakabatwa. Shandisa chidziviriro pakucheka.",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)":
        "Shandisa fungicide ine copper. Vandudza mhepo mumunda wemizambiringa. Bvisa mashizha akabatwa nokukasika.",
    "Orange - Haunglongbing (Citrus Greening)":
        "Hapana mushonga unorapa. Bvisa uye uparadze miti yakabatwa nokukasika kudzivirira kupararira. Dzvanya tuvhuvhu twe Asian citrus psyllid netumushonga twemhuka.",
    "Peach - Bacterial Spot":
        "Shandisa bactericide ine copper kubva pakutanga kwechirimo. Dzivisa kudiridza kunopfuura. Sima mhando dzine kudzivirira.",
    "Pepper - Bacterial Spot":
        "Shandisa bactericide ine copper. Dzivisa kushanda nezvirimwa zvakanyorova. Shandisa mbeu dzakachena dzisina chirwere.",
    "Potato - Early Blight":
        "Shandisa chlorothalonil kana mancozeb fungicide. Ona kuti zvirimwa zvine zvokudya zvakakwana — zvirimwa zvisina simba ndizvo zvinokurumidza kubatwa. Chinja zvirimwa.",
    "Potato - Late Blight":
        "Shandisa metalaxyl kana cymoxanil fungicide nokukasika. Paradza zvose zvakabatwa. Usasiya mbatatisi dzakabatwa muvhu.",
    "Squash - Powdery Mildew":
        "Shandisa potassium bicarbonate kana sulfur fungicide. Vandudza mhepo pakati pezvirimwa. Dzivisa kudiridza kunopfuura.",
    "Strawberry - Leaf Scorch":
        "Bvisa mashizha akabatwa. Shandisa fungicide (captan kana myclobutanil). Chengetedza kuyerera kwemvura nemhepo.",
    "Tomato - Bacterial Spot":
        "Shandisa bactericide ine copper. Bvisa marara ezvirimwa zvakabatwa. Dzivisa kudiridza kunopfuura uye kushanda nezvirimwa zvakanyorova.",
    "Tomato - Early Blight":
        "Shandisa chlorothalonil fungicide pakuona kwekutanga. Isa mulch kuti udzivirira mvura yevhu kusvika pamashizha. Bvisa mashizha akabatwa ari pasi.",
    "Tomato - Late Blight":
        "Shandisa fungicide ine copper nokukasika. Bvisa uye uparadze zvose zvakabatwa. Dzivisa kudiridza kunopfuura.",
    "Tomato - Leaf Mold":
        "Vandudza mhepo muhouse. Shandisa mancozeb kana chlorothalonil fungicide. Deredza unyoro hunopfuura 85%.",
    "Tomato - Septoria Leaf Spot":
        "Shandisa chlorothalonil fungicide. Bvisa mashizha akabatwa. Chinja zvirimwa uye dzivisa kudiridza kunopfuura.",
    "Tomato - Spider Mites (Two-spotted Spider Mite)":
        "Shandisa miticide (abamectin kana spiromesifen). Pfuvhira pasi pemashizha zvakanaka. Wedzera unyoro — spider mites dzinofarira kuoma.",
    "Tomato - Target Spot":
        "Shandisa chlorothalonil kana mancozeb fungicide. Bvisa marara akabatwa. Vandudza mhepo.",
    "Tomato - Yellow Leaf Curl Virus":
        "Hapana mushonga unorapa — bvisa uye uparadze zvirimwa zvakabatwa nokukasika. Dzvanya tuvhuvhu twewhitefly netumushonga twemhuka. Shandisa mhando dzine kudzivirira.",
    "Tomato - Mosaic Virus":
        "Hapana mushonga unorapa — bvisa uye uparadze zvirimwa zvakabatwa. Geza maoko nezvishandiso zvakanaka. Dzvanya tuvhuvhu twe aphid.",
}

# ── Treatment recommendations — isiNdebele (nd) ───────────────────────────────
TREATMENTS_ND = {
    "Apple - Apple Scab":
        "Sebenzisa umuthi we fungicide (captan kumbe myclobutanil) ngesikhathi sokuqala kuhluma. Qoqa njalo utshabalalise amaqabunga awileyo. Tshala uhlobo oluvikelekile.",
    "Apple - Black Rot":
        "Gunda njalo utshabalalise amagatsha athintekile. Sebenzisa i-fungicide ele-copper. Susa izithelo ezomileyo esihlahleni.",
    "Apple - Cedar Apple Rust":
        "Sebenzisa i-fungicide yokuvikela entwasahlobo. Susa izihlahla ze-cedar/juniper eziseduzane uma kungenzeka. Sebenzisa uhlobo oluvikelekile.",
    "Cherry - Powdery Mildew":
        "Sebenzisa i-sulfur kumbe i-potassium bicarbonate fungicide. Thuthukisa umoya ngokuquma. Gwema i-fertilizer ele-nitrogen elinengi.",
    "Corn - Cercospora Leaf Spot (Gray Leaf Spot)":
        "Sebenzisa i-fungicide ngesikhathi sokuqhakaza. Tshala uhlobo oluvikelekile. Phendukisa izithombo minyaka yonke.",
    "Corn - Common Rust":
        "Sebenzisa i-fungicide masinyane uma kubonakala okuqala. Tshala uhlobo oluvikela i-rust. Hlola amasimu njalo ngesikhathi sokushisa nokuswakama.",
    "Corn - Northern Leaf Blight":
        "Sebenzisa i-fungicide ngesikhathi sokuqhakaza uma isifo sibonakele. Sebenzisa uhlobo oluvikelekile. Phendukisa izithombo.",
    "Grape - Black Rot":
        "Sebenzisa i-mancozeb kumbe i-myclobutanil fungicide kusukela ekuhlumeni. Susa zonke izithelo ezomileyo. Gcina umoya omuhle.",
    "Grape - Black Measles (Esca)":
        "Akukho umuthi owelaphayo. Susa njalo utshabalalise izigodo ezithintekile. Sebenzisa isivikelo emva kokuquma.",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)":
        "Sebenzisa i-fungicide ele-copper. Thuthukisa umoya esimini samagilebhisi. Susa amaqabunga athintekile masinyane.",
    "Orange - Haunglongbing (Citrus Greening)":
        "Akukho umuthi owelaphayo. Susa njalo utshabalalise izihlahla ezithintekile masinyane ukuvikela ukusabalala. Lawula i-Asian citrus psyllid ngomuthi wezinambuzane.",
    "Peach - Bacterial Spot":
        "Sebenzisa i-bactericide ele-copper kusukela entwasahlobo. Gwema ukunisela okugcwele. Tshala uhlobo oluvikelekile.",
    "Pepper - Bacterial Spot":
        "Sebenzisa i-bactericide ele-copper. Gwema ukusebenza nezithombo ezimanzi. Sebenzisa imbewu ehlanzekileyo engenasifo.",
    "Potato - Early Blight":
        "Sebenzisa i-chlorothalonil kumbe i-mancozeb fungicide. Qinisekisa ukuthi izithombo zilukolo olwanele — izithombo ezibuthakathaka ziba lengozi enkulu. Phendukisa izithombo.",
    "Potato - Late Blight":
        "Sebenzisa i-metalaxyl kumbe i-cymoxanil fungicide masinyane. Tshabalalisa konke okuthintekile. Ungatshiyi amagwili athintekile emhlabathini.",
    "Squash - Powdery Mildew":
        "Sebenzisa i-potassium bicarbonate kumbe i-sulfur fungicide. Thuthukisa umoya phakathi kwezithombo. Gwema ukunisela okugcwele.",
    "Strawberry - Leaf Scorch":
        "Susa amaqabunga athintekile. Sebenzisa i-fungicide (captan kumbe myclobutanil). Gcina ukugeleza kwamanzi nomoya.",
    "Tomato - Bacterial Spot":
        "Sebenzisa i-bactericide ele-copper. Susa imfucumfucu yezithombo ezithintekile. Gwema ukunisela okugcwele nokusebenza nezithombo ezimanzi.",
    "Tomato - Early Blight":
        "Sebenzisa i-chlorothalonil fungicide ngokubona kokuqala. Faka i-mulch ukuvikela amanzi enhlabathi afinyelela emaqabungeni. Susa amaqabunga athintekile aphansi.",
    "Tomato - Late Blight":
        "Sebenzisa i-fungicide ele-copper masinyane. Susa njalo utshabalalise konke okuthintekile. Gwema ukunisela okugcwele.",
    "Tomato - Leaf Mold":
        "Thuthukisa umoya endlini yokuhlanyela. Sebenzisa i-mancozeb kumbe i-chlorothalonil fungicide. Nciphisa ubumanzi obungaphansi kuka-85%.",
    "Tomato - Septoria Leaf Spot":
        "Sebenzisa i-chlorothalonil fungicide. Susa amaqabunga athintekile. Phendukisa izithombo njalo ugweme ukunisela okugcwele.",
    "Tomato - Spider Mites (Two-spotted Spider Mite)":
        "Sebenzisa i-miticide (abamectin kumbe spiromesifen). Chela ngezansi kwamaqabunga ngokuphelele. Yandisa ubumanzi — i-spider mites zithanda okomileyo.",
    "Tomato - Target Spot":
        "Sebenzisa i-chlorothalonil kumbe i-mancozeb fungicide. Susa imfucumfucu ethintekile. Thuthukisa umoya.",
    "Tomato - Yellow Leaf Curl Virus":
        "Akukho umuthi owelaphayo — susa njalo utshabalalise izithombo ezithintekile masinyane. Lawula i-whitefly ngomuthi wezinambuzane. Sebenzisa uhlobo oluvikelekile.",
    "Tomato - Mosaic Virus":
        "Akukho umuthi owelaphayo — susa njalo utshabalalise izithombo ezithintekile. Geza izandla nezinto zokusebenza kuhle. Lawula i-aphid.",
}

LANGUAGES = {
    "English": TREATMENTS,
    "Shona": TREATMENTS_SN,
    "isiNdebele": TREATMENTS_ND,
}

@st.cache_resource
def load_model():
    return tf.saved_model.load("crop_disease_savedmodel")

# ── Predict function ──────────────────────────────────────────────────────────
def predict(img: Image.Image):
    img_resized = img.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, 0)
    model = load_model()
    infer = model.signatures["serving_default"]
    output_key = list(infer.structured_outputs.keys())[0]
    preds = infer(tf.constant(arr))[output_key].numpy()[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    return preds, top3_idx

# ── Confidence bar chart ──────────────────────────────────────────────────────
def confidence_chart(preds, top3_idx):
    names  = [CLASS_NAMES[i].split(" - ")[-1] for i in top3_idx]
    scores = [round(float(preds[i]) * 100, 1) for i in top3_idx]
    colors = ["#1D9E75" if i == 0 else "#B4B2A9" for i in range(3)]

    fig, ax = plt.subplots(figsize=(5, 2))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.tick_params(labelsize=9)
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{score}%", va="center", fontsize=9, color="#444441")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    st.markdown("## Crop Disease Detector")
    st.markdown("**Malta Academy** — STEM Symposium")
    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload a leaf image",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear photo of the affected leaf. PNG or JPG."
    )
    st.markdown("---")
    language = st.selectbox(
        "Treatment language",
        options=list(LANGUAGES.keys()),
        index=0,
        help="Choose the language for treatment recommendations."
    )
    st.markdown("---")
    st.markdown("**Model:** MobileNetV2 (transfer learning)")
    st.markdown("**Dataset:** PlantVillage (54,306 images)")
    st.markdown("**Classes:** 38 disease categories")
    st.markdown("**Accuracy:** ~92% validation")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload a PNG/JPG of a leaf")
    st.markdown("2. Wait 2–3 seconds for prediction")
    st.markdown("3. Read the disease name and treatment")

# Main area
st.title("Crop Disease Detection System")
st.markdown("AI-powered plant disease diagnosis using MobileNetV2 deep learning — *offline, instant, and free*.")

if uploaded is None:
    st.info("Upload a leaf image in the sidebar to begin diagnosis.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Crop species supported", "14")
    with col2:
        st.metric("Disease classes", "38")
    with col3:
        st.metric("Model accuracy", "~92%")

else:
    img = Image.open(uploaded)
    preds, top3_idx = predict(img)
    top_idx    = top3_idx[0]
    top_label  = CLASS_NAMES[top_idx]
    top_conf   = round(float(preds[top_idx]) * 100, 1)
    is_healthy = top_label in HEALTHY_CLASSES
    treatment  = LANGUAGES[language].get(top_label, None)

    # ── Result header ─────────────────────────────────────────────────────────
    col_res, col_img = st.columns([2, 1])

    with col_res:
        if is_healthy:
            st.success(f"### {top_label}")
            st.markdown(f"**Confidence: {top_conf}%** — this plant appears healthy.")
        else:
            st.error(f"### {top_label}")
            st.markdown(f"**Confidence: {top_conf}%** — disease detected.")

        # Metric cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Top prediction", top_label.split(" - ")[-1])
        m2.metric("Confidence", f"{top_conf}%")
        m3.metric("Status", "Healthy" if is_healthy else "Disease detected")

        # Confidence bar chart
        st.markdown("#### Top 3 predictions")
        st.pyplot(confidence_chart(preds, top3_idx))

    with col_img:
        st.markdown("#### Uploaded leaf")
        st.image(img, use_column_width=True)
        st.caption(f"File: {uploaded.name}")

    st.markdown("---")

    # ── Treatment ─────────────────────────────────────────────────────────────
    if is_healthy:
        st.success("No treatment needed — this plant is healthy.")
    elif treatment:
        st.warning(f"**Recommended treatment for {top_label}:**\n\n{treatment}")
    else:
        fallback_msgs = {
            "English": "Consult a local agronomist for specific treatment advice for this disease.",
            "Shona": "Bvunza nyanzvi yezvirimwa yepedyo kuti uwane mazano ekurapa chirwere ichi.",
            "isiNdebele": "Buza isazi sezithombo eseduze ukuthola iseluleko sokwelapha lesi sifo."
        }
        st.info(fallback_msgs[language])
    st.markdown("---")

    # ── Full probability table ────────────────────────────────────────────────
    with st.expander("View full prediction probabilities (all 38 classes)"):
        all_probs = sorted(
            [(CLASS_NAMES[i], round(float(preds[i]) * 100, 2)) for i in range(len(CLASS_NAMES))],
            key=lambda x: x[1], reverse=True
        )
        import pandas as pd
        df = pd.DataFrame(all_probs, columns=["Disease / Class", "Confidence (%)"])
        st.dataframe(df, use_container_width=True, hide_index=True)