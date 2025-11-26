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

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="NarrativeNexus | AI Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Decorated" Look
st.markdown("""
    <style>
    /* Global Font & Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Card-like containers */
    .stMarkdown, .stBlock {
        border-radius: 10px;
    }
    
    /* Header Styling */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 800;
        background: -webkit-linear-gradient(#1A2980, #26D0CE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    h2, h3 {
        color: #34495e;
    }
    
    /* Custom container for file results */
    .file-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #26D0CE;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Caching Resources ---
@st.cache_resource
def load_resources():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('stopwords')
    # Load summarizer (lightweight version)
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_resources()
lemmatizer = WordNetLemmatizer()

# --- 3. Processing Functions ---

def preprocess_for_gensim(text):
    """
    Applies the preprocessing logic from your notebook:
    - simple_preprocess (tokenize & de-accent)
    - Remove Stopwords
    - Remove short words (< 3 chars)
    - Lemmatize
    """
    tokens = simple_preprocess(text, deacc=True)
    clean_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in STOPWORDS and len(token) > 3
    ]
    return clean_tokens

def run_gensim_lda(clean_docs_list, num_topics=5):
    """
    Runs the LDA model using Gensim.
    """
    # Create Dictionary
    id2word = corpora.Dictionary(clean_docs_list)
    
    # Create Corpus (Term Document Frequency)
    corpus = [id2word.doc2bow(text) for text in clean_docs_list]
    
    # Train Model
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

# --- 4. Main Application Logic ---

# Header Section
st.title("✨ NarrativeNexus AI")
st.markdown("#### Advanced NLP Pipeline: Topic Modeling, Sentiment & Summarization")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    num_topics = st.slider("Number of LDA Topics", 2, 10, 3)
    st.info("Upload your .txt files to generate insights.")
    st.markdown("---")
    st.caption("Built with Streamlit & Gensim")

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
        # Preprocess for Gensim (list of tokens)
        processed_docs_for_lda.append(preprocess_for_gensim(text))

    # --- GLOBAL ANALYSIS: TOPIC MODELING ---
    st.subheader("🔍 Corpus-Wide Topic Modeling")
    st.markdown("Discovering hidden themes across all uploaded documents...")
    
    with st.spinner("Training Gensim LDA Model..."):
        try:
            lda_model, corpus, id2word = run_gensim_lda(processed_docs_for_lda, num_topics=num_topics)
            
            # Display Topics in Columns
            cols = st.columns(num_topics)
            for idx, topic in lda_model.print_topics(-1):
                # Parse string '0.05*"word" + ...'
                topic_clean = topic.split('+')
                words = []
                weights = []
                for term in topic_clean:
                    weight, word = term.split('*')
                    words.append(word.strip().replace('"', ''))
                    weights.append(float(weight))
                
                # Visualize Topic
                with cols[idx % num_topics]:
                    with st.container(border=True):
                        st.markdown(f"**Topic {idx+1}**")
                        fig = px.bar(x=weights[:5], y=words[:5], orientation='h', 
                                     labels={'x':'Weight', 'y':''},
                                     color=weights[:5], color_continuous_scale='Bluyl')
                        fig.update_layout(showlegend=False, height=200, margin=dict(l=0,r=0,t=0,b=0), yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.warning(f"LDA Error (Data might be too sparse): {e}")

    st.markdown("---")

    # --- INDIVIDUAL FILE ANALYSIS ---
    st.subheader("📄 Document Insights")
    
    # Create tabs for cleaner navigation
    tabs = st.tabs([f"{name}" for name in filenames])

    for i, tab in enumerate(tabs):
        with tab:
            # Layout: 2 Columns
            col_left, col_right = st.columns([1.5, 1])
            
            # Analysis Logic
            polarity, sentiment_label = get_sentiment(raw_texts[i])
            
            with col_left:
                st.markdown("### 📝 Summary")
                with st.spinner("Summarizing..."):
                    try:
                        # Truncate for speed/safety
                        summary_res = summarizer(raw_texts[i][:1500], max_length=130, min_length=30, do_sample=False)
                        st.info(summary_res[0]['summary_text'])
                    except:
                        st.warning("Text too short or complex to summarize.")

                st.markdown("### 📜 Raw Text")
                with st.expander("Click to read full text"):
                    st.write(raw_texts[i])

            with col_right:
                st.markdown("### 📊 Sentiment")
                
                # Color coding for sentiment
                color = "green" if polarity > 0.1 else "red" if polarity < -0.1 else "gray"
                
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = polarity,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"<b>{sentiment_label}</b>", 'font': {'size': 24}},
                    delta = {'reference': 0},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [-1, -0.1], 'color': "mistyrose"},
                            {'range': [-0.1, 0.1], 'color': "whitesmoke"},
                            {'range': [0.1, 1], 'color': "honeydew"}],
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=30,r=30,t=50,b=30))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("### ☁️ Word Cloud")
                try:
                    wc = WordCloud(background_color='white', width=400, height=300, colormap='viridis').generate(raw_texts[i])
                    fig_wc, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig_wc)
                except:
                    st.caption("Insufficient data for Word Cloud")

else:
    # Welcome Screen / Empty State
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; background-color: white; border-radius: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
        <h2 style="color: #2c3e50;">Welcome to NarrativeNexus</h2>
        <p style="color: gray; font-size: 1.2rem;">Upload your text documents on the left to unlock AI-powered insights.</p>
    </div>
    """, unsafe_allow_html=True)
