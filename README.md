# RNN-based-Question-Answer-System

This is a **simple PyTorch-based RNN Question-Answer system** with a **modern Streamlit UI**. It predicts a **single-word answer** for a given input question using a trained Recurrent Neural Network (RNN).

---

## 🚀 Features

- Simple **RNN architecture** implemented in PyTorch.
- Predicts answers for single-word questions.
- **Modern Streamlit UI** with:
  - Custom background and styling
  - Input box with placeholder
  - Stylish "Get Answer" button
  - Answer display box with confidence handling
- Handles unknown words with `<UNK>` token.
- Includes caching for faster resource loading (`@st.cache_resource`).

---

## 🛠️ Requirements

- Python 3.10+
- Libraries:
  ```bash
  pip install streamlit torch
📂 Project Structure
RNN_QA/
├── qa_app.py        # Main Streamlit app
├── model.pth        # Trained RNN model weights (PyTorch)
├── vocab.pth        # Vocabulary dictionary (PyTorch)
├── README.md        # Project documentation
└── requirements.txt # (Optional) Python dependencies

<h3>📝 How It Works</h3>
<ol><li>
Model and Utilities</li>
<ul><li>
SimpleRNN class: Embedding → RNN → Linear layer</li>
<li>
tokenize(text): Lowercases and removes punctuation</li>
<li>
text_to_indices(text, vocab): Converts tokens to numerical indices</li>
<li>
predict(model, question, vocab, index_to_word): Generates answer for input question</li>
</ul>
<li>
Streamlit UI
</li>
<ul><li>
Wide layout with custom CSS styling
</li>
<li>
Input box for question
</li>
<li>
"Get Answer" button triggers prediction
</li>
<li>
Displays predicted answer in a styled box
</li>
<li>
Handles invalid or empty input
</li>
</ul>
<li>
Caching
</li>
<ul><li>
load_resources() uses @st.cache_resource to load model and vocab only once for faster response.
</li></ul>
<h2>⚡ How to Run</h2>

<b>Clone this repository:</b>
<ol>
<li>
git clone <your-repo-url></li>
<li>
cd RNN_QA</li>

<li>
Install required libraries:</li>
<li>
pip install -r requirements.txt
Ensure model.pth and vocab.pth exist in the same directory.</li>
<li>
Run the Streamlit app:</li>
<li>
streamlit run qa_app.py</li>

<li>
Open the provided URL in your browser (usually http://localhost:8501).</li>
</ol>

<h3>💡 Notes</h3>
<ul>
<li>
This system is designed for single-word answers.</li>
<li>
For multi-word answers or larger datasets, the model and preprocessing will need modifications.</li>
<li>
Streamlit caching ensures that the model and vocabulary are loaded only once, improving app responsiveness.</li></ul>

<h3>📚 References</h3>
<ul>
<li>
<a href="https://chatgpt.com/c/68e0cfec-8180-8320-920b-8a3800a90888#:~:text=PyTorch%20Documentation">PyTorch Documentation</a>
</li>
<li>
<a href="https://docs.streamlit.io/">Streamlit Documentation</a>
</li>
<li>
<a href="https://chatgpt.com/c/68e0cfec-8180-8320-920b-8a3800a90888#:~:text=Basic%20RNN%20implementation%20concepts%3A%20Understanding%20RNNs">Basic RNN implementation concepts: Understanding RNNs</a>
</li>
</ul>
<h3>🧑‍💻 Author</h3>
<ul>
<li>
Developed By: <a href="https://www.linkedin.com/in/dheeraj-saroha/">Dheeraj Saroha</a>
</li>
<li>
Email: dheerajsaroha2892002@gmail.com</a>
</li>
