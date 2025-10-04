# qa_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os

# --- 1. Model and Utility Functions (Modified from model.py) ---

# Define the paths for your saved model and vocab
MODEL_PATH = 'model.pth' 
VOCAB_PATH = 'vocab.pth'

def tokenize(text):
    """Converts text to lowercase and removes punctuation."""
    if not text:
        return []
    text = text.lower()
    text = text.replace("?", '')
    text = text.replace("'", "")
    return text.split()

def text_to_indices(text, vocab):
    """Converts a tokenized list of words to numerical indices."""
    indexed_text = []
    for token in tokenize(text):
        indexed_text.append(vocab.get(token, vocab['<UNK>']))
    return indexed_text

# Define the model architecture
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)
    
    def forward(self, question):
        embedded_question = self.embedding(question)
        _, final = self.rnn(embedded_question)
        output = self.fc(final.squeeze(0))
        return output

# Use Streamlit's caching mechanism to load the model and vocab once
@st.cache_resource
def load_resources():
    """Loads the vocabulary and model state dictionary."""
    
    # 1. Load Vocab
    if not os.path.exists(VOCAB_PATH):
        st.error(f"❌ ERROR: Vocabulary file '{VOCAB_PATH}' not found. Please create it first.")
        return None, None
    try:
        vocab = torch.load(VOCAB_PATH)
        VOCAB_SIZE = len(vocab)
        index_to_word = {i: word for word, i in vocab.items()}
    except Exception as e:
        st.error(f"❌ ERROR: Failed to load vocabulary from '{VOCAB_PATH}': {e}")
        return None, None

    # 2. Initialize and Load Model
    model = SimpleRNN(VOCAB_SIZE)
    if not os.path.exists(MODEL_PATH):
        st.warning(f"⚠️ WARNING: Model weights file '{MODEL_PATH}' not found. Using untrained weights.")
    else:
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except Exception as e:
            st.error(f"❌ ERROR: Failed to load model weights from '{MODEL_PATH}': {e}")
            return None, None
            
    model.eval() # Set model to evaluation mode
    return model, vocab, index_to_word

# Prediction function
def predict(model, question, vocab, index_to_word, threshold=0.5):
    """Predicts the answer for a given question."""
    
    numerical_question = text_to_indices(question, vocab)
    
    if not numerical_question:
        return "Please ask a valid question."
    
    # Add batch dimension (Batch_size=1, Seq_len)
    question_tensor = torch.tensor(numerical_question, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        output = model(question_tensor)

    # Convert logits to probabilities
    probs = F.softmax(output, dim=1)
    
    # Find the index of the max probability
    value, index = torch.max(probs, dim=1)
    
    predicted_word = index_to_word.get(index.item(), '<UNK>')

    # Confidence check
    if value.item() < threshold:
        # return f"I don't know (Confidence: {value.item():.2f})."
        return f"I don't know."
    
    # Capitalize the predicted word for a cleaner presentation
    return predicted_word.capitalize()

# --- 2. Streamlit UI/UX ---

# Set a wide layout and a nice page title
st.set_page_config(
    page_title="RNN Question-Answer System", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI/UX
st.markdown("""
<style>
.main {
    background-color: #f0f2f6; /* Light gray background */
}
.stButton>button {
    background-color: #4CAF50; /* Primary Green */
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #43A047; /* Darker green on hover */
}
.answer-box {
    background-color: #e8f5e9; /* Very light green */
    border-left: 5px solid #4CAF50; /* Green border */
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
}
.stTextInput>div>div>input {
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# Load resources
model, vocab, index_to_word = load_resources()

# --- Application Title and Description ---
st.title("🧠 Simple PyTorch RNN Question Answering System")
st.markdown("---")
st.markdown(
    """
    <p class='subtitle'>
    This application uses a trained Recurrent Neural Network (RNN) to predict a single-word answer 
    based on your input question.
    </p>
    """, unsafe_allow_html=True
)

if model is not None and vocab is not None:
    
    # --- Input Area ---
    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.text_input(
            "**Ask a Question**", 
            placeholder="e.g., What is the capital of France?",
            key="question_input"
        )

    with col2:
        # UX: Use st.empty() to align the button
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("Get Answer", key="ask_button"):
            if question:
                with st.spinner('Model is analyzing the question...'):
                    # Call the prediction function
                    answer = predict(model, question, vocab, index_to_word, threshold=0.5)
                
                # Store the result for display
                st.session_state['last_answer'] = answer
                st.session_state['last_question'] = question
            else:
                st.session_state['last_answer'] = "Please enter a question to get a response."
                st.session_state['last_question'] = ""


    # --- Output Area ---
    if 'last_answer' in st.session_state and st.session_state['last_answer']:
        st.markdown("## Predicted Answer")
        st.markdown(
            f"""
            <div class='answer-box'>
                <p style='font-size: 1.1em; margin: 0;'>
                <strong style='color:#388E3C;'>Your Question: {st.session_state['last_question']}</strong>
                </p>
                <p style='font-size: 1.5em; font-weight: bold; color: #388E3C; margin: 5px 0 0 0;'>
                {st.session_state['last_answer']}
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    
else:
    st.error("Application setup failed. Please check the console for errors regarding missing or corrupt `model.pth` or `vocab.pth` files.")