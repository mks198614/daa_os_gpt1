import streamlit as st
from transformers import pipeline

# Load the model only once using session state
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

model = load_model()

# Title
st.title("🤖 DAA & OS GPT - Free Tutor")
st.subheader("Ask anything related to Design and Analysis of Algorithms or Operating Systems.")

# Input box
user_input = st.text_area("💬 Ask a question", height=100)

# If input is given
if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        prompt = (
            "You are an expert tutor for B.Tech students in Design and Analysis of Algorithms and Operating Systems.\n"
            f"Q: {user_input}\nA:"
        )
        with st.spinner("Thinking..."):
            response = model(prompt, max_length=150, do_sample=True, temperature=0.7)[0]["generated_text"]
            answer = response.split("A:")[-1].strip()
            st.success("Answer:")
            st.write(answer)
