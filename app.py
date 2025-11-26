import streamlit as st
import pandas as pd
import re
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="NarrativeNexus | AI Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Dark Mode & Decorated CSS ---
st.markdown("""
    <style>
    /* --- GLOBAL SETTINGS --- */
    .stApp {
        /* Deep Midnight Gradient Background */
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        color: #e0e0e0;
    }
    
    /* --- TYPOGRAPHY --- */
    h1 {
        font-weight: 900;
        /* Neon Gradient Text */
        background: -webkit-linear-gradient(0deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 10px rgba(0, 210, 255, 0.3);
    }
    h2, h3 {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    p, label, .stMarkdown {
        color: #d1d1d1 !important;
    }

    /* --- SIDEBAR STYLING --- */
    section[data-testid="stSidebar"] {
        background-color: #0a1117; /* Very dark blue/black */
        border-right: 1px solid #333;
    }
    
    /* --- METRIC CARDS --- */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    div[data-testid="stMetricLabel"] {
        color: #00d2ff !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* --- FILE UPLOADER --- */
    div[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px dashed #4a4a4a;
    }

    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.05);
        border-radius: 5px;
        color: white;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 210, 255, 0.2) !important;
        border-bottom: 2px solid #00d2ff;
        color: #fff !important;
    }
    
    /* --- EXPANDER --- */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05);
        color: white !important;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Caching Resources ---
@st.cache_resource
def load_resources():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('stopwords')
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_resources()
lemmatizer = WordNetLemmatizer()

# --- 4. Processing Functions ---

def preprocess_for_gensim(text):
    tokens = simple_preprocess(text, deacc=True)
    clean_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in STOPWORDS and len(token) > 3
    ]
    return clean_tokens

def run_gensim_lda(clean_docs_list, num_topics=5):
    id2word = corpora.Dictionary(clean_docs_list)
    corpus = [id2word.doc2bow(text) for text in clean_docs_list]
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    return lda_model, corpus, id2word

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1: sentiment = 'Positive'
    elif polarity < -0.1: sentiment = 'Negative'
    else: sentiment = 'Neutral'
    return polarity, sentiment

# --- 5. Main Application Logic ---

# Header Section
st.title("✨ NarrativeNexus AI")
st.markdown("<h4 style='text-align: center; color: #b0c4de;'>Dark Mode NLP Pipeline: Topic Modeling, Sentiment & Summarization</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    num_topics = st.slider("Number of LDA Topics", 2, 10, 4)
    st.info("Upload your .txt files to generate insights.")
    st.markdown("---")
    st.caption("Powered by Streamlit & Gensim")

# File Uploader
uploaded_files = st.file_uploader("📂 Upload Dataset (.txt)", type="txt", accept_multiple_files=True)

if uploaded_files:
    # --- Data Preparation ---
    raw_texts = []
    filenames = []
    processed_docs_for_lda = []

    for file in uploaded_files:
        text = file.getvalue().decode("utf-8")
        raw_texts.append(text)
        filenames.append(file.name)
        processed_docs_for_lda.append(preprocess_for_gensim(text))

    # --- GLOBAL ANALYSIS: TOPIC MODELING ---
    st.subheader("🔍 Corpus-Wide Topic Modeling")
    st.markdown("Discovering hidden themes across all uploaded documents...")
    
    with st.spinner("Training Gensim LDA Model..."):
        try:
            lda_model, corpus, id2word = run_gensim_lda(processed_docs_for_lda, num_topics=num_topics)
            
            cols = st.columns(num_topics)
            for idx, topic in lda_model.print_topics(-1):
                # Parse string data
                topic_clean = topic.split('+')
                words = []
                weights = []
                for term in topic_clean:
                    weight, word = term.split('*')
                    words.append(word.strip().replace('"', ''))
                    weights.append(float(weight))
                
                # Visualize Topic
                with cols[idx % num_topics]:
                    # Helper to make container background transparent/dark-friendly
                    with st.container(border=True):
                        st.markdown(f"<h4 style='text-align:center; color: #00d2ff;'>Topic {idx+1}</h4>", unsafe_allow_html=True)
                        
                        # Updated Plotly Chart for Dark Mode
                        fig = px.bar(x=weights[:5], y=words[:5], orientation='h', 
                                     labels={'x':'Weight', 'y':''},
                                     color=weights[:5], 
                                     color_continuous_scale='Tealgrn') # 'Tealgrn' looks good on dark
                        
                        fig.update_layout(
                            showlegend=False, 
                            height=200, 
                            margin=dict(l=0,r=0,t=0,b=0), 
                            yaxis={'categoryorder':'total ascending'},
                            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
                            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                            font=dict(color="white"),       # White text for axes
                            xaxis=dict(showgrid=False),
                            coloraxis_showscale=False       # Hide color bar
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.warning(f"LDA Error (Data might be too sparse): {e}")

    st.markdown("---")

    # --- INDIVIDUAL FILE ANALYSIS ---
    st.subheader("📄 Document Insights")
    
    tabs = st.tabs([f"{name}" for name in filenames])

    for i, tab in enumerate(tabs):
        with tab:
            col_left, col_right = st.columns([1.5, 1])
            
            # Analysis Logic
            polarity, sentiment_label = get_sentiment(raw_texts[i])
            
            with col_left:
                st.markdown("### 📝 Summary")
                with st.spinner("Summarizing..."):
                    try:
                        summary_res = summarizer(raw_texts[i][:1500], max_length=130, min_length=30, do_sample=False)
                        # Custom styled summary box
                        st.markdown(f"""
                        <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #00d2ff;">
                            <p style="color: white !important; margin: 0;">{summary_res[0]['summary_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except:
                        st.warning("Text too short or complex to summarize.")

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📜 Read Full Source Text"):
                    st.write(raw_texts[i])

            with col_right:
                st.markdown("### 📊 Sentiment")
                
                # Color coding
                color = "#00d2ff" if polarity > 0.1 else "#ff4b4b" if polarity < -0.1 else "#ffa500"
                
                # Gauge Chart (Dark Mode Optimized)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = polarity,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"<span style='color:white'>{sentiment_label}</span>", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [-1, 1], 'tickcolor': "white", 'tickwidth': 2},
                        'bar': {'color': color},
                        'bgcolor': "rgba(255,255,255,0.1)",
                        'borderwidth': 2,
                        'bordercolor': "white",
                    }
                ))
                fig_gauge.update_layout(
                    height=220, 
                    margin=dict(l=30,r=30,t=50,b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Word Cloud (Dark Mode Optimized)
                st.markdown("### ☁️ Word Cloud")
                try:
                    # Black background wordcloud to blend with dark theme
                    wc = WordCloud(background_color='#0f2027', width=400, height=250, colormap='GnBu').generate(raw_texts[i])
                    fig_wc, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    # Make matplotlib figure transparent
                    fig_wc.patch.set_alpha(0) 
                    st.pyplot(fig_wc)
                except:
                    st.caption("Insufficient data for Word Cloud")

else:
    # Empty State Decoration
    st.markdown("""
    <br><br>
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; opacity: 0.7;">
        <h1 style="font-size: 4rem;">📂</h1>
        <h3 style="color: white;">Awaiting Data...</h3>
        <p style="color: gray;">Upload text files to the sidebar to activate the Neural Engine.</p>
    </div>
    """, unsafe_allow_html=True)
