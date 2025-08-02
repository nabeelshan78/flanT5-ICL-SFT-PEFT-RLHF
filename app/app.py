import streamlit as st
from utils import load_model, generate_summary
import os
import time

st.set_page_config(
    page_title="Dialogue Summarization with Advanced FLAN-T5 Models",
    layout="wide",
    initial_sidebar_state="expanded", 
)

# --- Pre-defined dialogues for quick testing ---
pre_canned_dialogues = {
    "Custom Dialogue": "",
    "Meeting Agenda Discussion": """
Amanda: Hey team, let's go over the agenda for today's meeting. We need to discuss Q3 results, the new marketing campaign, and the project timeline for the new feature launch.
John: Sounds good. I have the Q3 numbers ready, but I'm still waiting on final approval for the marketing budget.
Amanda: Okay, we can address that. Nabeel, are the mockups for the new feature launch ready?
Nabeel: Yes, I've just uploaded them to the shared drive. I'll walk everyone through them.
Amanda: Perfect. Let's start with the Q3 results then.
""",
    "Customer Support Issue Resolution": """
Customer: Hi, I'm having trouble with my account. My payment for the subscription didn't go through, and now I can't access any features.
Agent: Hello! I'm sorry to hear that. Can you please confirm the email address associated with your account?
Customer: It's jsmith@example.com.
Agent: Thank you. I see the issue. There was a problem with the payment gateway. I've reset the payment link for you. Please try again, and let me know if it works.
Customer: Okay, it worked! Thank you so much for your help.
Agent: You're welcome! Is there anything else I can assist you with today?
""",
    "Software Project Update": """
Chris: Hey Antje, a quick update on the new feature. We've completed the initial design phase, and the development team is ready to start coding.
Antje: That's great news, Chris! What's the estimated timeline for the development phase?
Chris: We're projecting about two weeks for the core functionality, followed by one week of testing. So, around three weeks total.
Antje: Excellent. I'll inform the stakeholders and get a sign-off on the timeline. Thanks for the heads-up.
""",
    "Team Brainstorming Session": """
Sarah: Alright team, let's brainstorm some ideas for our new product's name. We want something catchy, memorable, and relevant to AI.
Mike: How about "Cognito"? It sounds smart and relates to cognition.
Emily: I like "Synapse AI" – it evokes connections and intelligence.
David: What about "Neuralink"? Oh wait, that's taken. How about "MindFlow"?
Sarah: Good ideas, everyone. Let's list them all and then vote. Cognito, Synapse AI, MindFlow. Any others?
Mike: "IntelliGrok"?
Emily: "AetherMind"?
Sarah: Okay, we have a good list to start with.
"""
}

# --- Helper function to load and cache models on demand ---
def load_and_cache_model_on_demand(model_key, model_type_str):
    """
    Loads a specific model if it's not already in session state, and caches it.
    Displays a spinner during loading.
    """
    if model_key not in st.session_state.loaded_models_data:
        with st.spinner(f"Loading `{model_key}` model... This might take a moment!"):
            model, tokenizer = load_model(model_type_str)
            st.session_state.loaded_models_data[model_key] = {"model": model, "tokenizer": tokenizer}
            
            success_placeholder = st.empty()
            success_placeholder.success(f"{model_key} loaded successfully!")
            time.sleep(2)  # duration in seconds
            success_placeholder.empty()  # clear the message

    return st.session_state.loaded_models_data[model_key]["model"], st.session_state.loaded_models_data[model_key]["tokenizer"]



# --- Initialize loaded models dictionary in session state ---
# This dictionary will store models as they are loaded
if "loaded_models_data" not in st.session_state:
    st.session_state.loaded_models_data = {}



# --- Initial Model Load (Full Fine-Tuned by default) ---
# This ensures only the default model is loaded when the app starts
DEFAULT_MODEL_KEY = "Full Fine-Tuned"
DEFAULT_MODEL_TYPE_STR = "full" 

# Load the default model at startup if it's not already loaded
if DEFAULT_MODEL_KEY not in st.session_state.loaded_models_data:
    load_and_cache_model_on_demand(DEFAULT_MODEL_KEY, DEFAULT_MODEL_TYPE_STR)



# --- Application Title and Introduction ---
st.title("💡 Dialogue Summarization with Advanced FLAN-T5 Models")
st.markdown(
    """
    Welcome to this interactive demonstration of **Large Language Models (LLMs)** for dialogue summarization!
    This application showcases the power of **FLAN-T5**, a prominent Transformer-based model,
    and compares its performance across different fine-tuning strategies:
    **Full Fine-Tuning**, **Parameter-Efficient Fine-Tuning (PEFT) with LoRA**, and the **Base Pretrained Model**.
    
    Select a model from the sidebar, choose an existing dialogue or enter your own, and generate a concise summary.
    """
)

st.markdown("---")


# --- Sidebar for Model Selection ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.markdown("Choose the FLAN-T5 model variant for summarization:")
    
    model_options = ["Full Fine-Tuned", "PEFT Fine-Tuned (LoRA)", "Base FLAN-T5 (Pretrained)"]
    default_index = model_options.index(DEFAULT_MODEL_KEY)

    model_choice = st.radio(
        "Select a Model:",
        model_options,
        index=default_index,
        key="model_radio_selection",
        help="""
        * **Full Fine-Tuned**: The entire model's parameters were updated during training on dialogue summarization data.
        * **PEFT Fine-Tuned (LoRA)**: Only a small fraction of the model's parameters were updated, making training more efficient while achieving comparable performance.
        * **Base FLAN-T5 (Pretrained)**: The original model, without any specific fine-tuning for dialogue summarization.
        """
    )



# --- Main Content Area (Columns) ---
col_dialogue, col_summary = st.columns([1.5, 1])

with col_dialogue:
    st.header("📝 Dialogue Input")
    
    # Dialogue Selection dropdown
    dialogue_choice = st.selectbox(
        "Choose an existing dialogue or enter your own:",
        list(pre_canned_dialogues.keys()),
        index=0,
        key="dialogue_selector", 
        help="Select a pre-defined conversation to quickly test the summarizer, or choose 'Custom Dialogue' to input your own."
    )

    # Initialize session state for user input if not present
    if "user_input_text_area" not in st.session_state:
        st.session_state.user_input_text_area = ""
    
    if dialogue_choice != st.session_state.get("last_selected_dialogue_for_update", "Custom Dialogue"):
        st.session_state.user_input_text_area = pre_canned_dialogues[dialogue_choice]
        st.session_state.last_selected_dialogue_for_update = dialogue_choice

    # Text area for dialogue input
    st.session_state.user_input_text_area = st.text_area(
        "Paste your conversation here:",
        value=st.session_state.user_input_text_area,
        height=300,
        placeholder="e.g., 'Person A: Hi, how are you? Person B: I'm good, thanks! How about you?'",
        key="dialogue_text_input" 
    )

    st.markdown("---")
    
    # Generate Summary button
    if st.button("✨ Generate Summary", use_container_width=True, type="primary"):
        if st.session_state.user_input_text_area.strip() == "":
            st.warning("Please enter some dialogue to summarize before generating.")
        else:
            # Determine the model_type_str based on the selected model_choice
            model_type_mapping = {
                "Full Fine-Tuned": "full",
                "PEFT Fine-Tuned (LoRA)": "peft",
                "Base FLAN-T5 (Pretrained)": "original"
            }
            selected_model_type_str = model_type_mapping[model_choice]
            model, tokenizer = load_and_cache_model_on_demand(model_choice, selected_model_type_str)
            
            with st.spinner(f"Generating summary using the {model_choice} model..."):
                summary = generate_summary(st.session_state.user_input_text_area, model, tokenizer)
                st.session_state['generated_summary'] = summary
                st.session_state['model_used_for_summary'] = model_choice
            st.success("Summary generated!")

with col_summary:
    st.header("📄 Model Generated Summary")
    
    if 'generated_summary' in st.session_state:
        st.info(f"Summary from **{st.session_state['model_used_for_summary']}** model:")
        st.text_area(
            "Output Summary:", 
            value=st.session_state['generated_summary'], 
            height=300, 
            key="summary_output_text_area" 
        )
    else:
        st.info("The generated summary will appear here after you click 'Generate Summary'.")

st.markdown("---")


# --- How it Works Section ---
st.header("🧠 How Dialogue Summarization Works")

# Short summary
st.markdown(
    """
    This app uses the **FLAN-T5 model**, a powerful Transformer-based architecture for text summarization.

    **How it works in short:**
    - **Encoder-Decoder:** Reads and rewrites input into a summary.
    - **Attention:** Focuses on key info while generating output.
    - **Tokenization:** Converts text to numbers (tokens).
    - **Fine-Tuning:** Trains the model for dialogue summarization (with full or PEFT using LoRA).
    - **Inference:** The model generates human-like summaries from input dialogues.
    """
)

# Full explanation in an expander
with st.expander("🔍 Learn more: Deep Dive into Transformers"):
    st.markdown(
        """
        At the heart of this application lies the **FLAN-T5 model**, a powerful variant of the **Transformer Neural Network**.
        Think of a Transformer as a super-smart assistant that can read, understand, and then write text,
        much like a child learning to read a story and then retell it in their own words.

        Here's a simplified breakdown of the magic behind it:

        * **1. Encoder-Decoder Architecture:**
            - **Encoder (The Reader):** Reads the entire dialogue and builds a numerical representation of its meaning.
            - **Decoder (The Writer):** Generates the summary based on the encoded information.

        * **2. Attention Mechanism (The Focus Button):**
            - Allows the model to selectively focus on the most relevant parts of the input.
            - **Self-Attention:** Understands internal relationships in the input.
            - **Encoder-Decoder Attention:** Helps generate each word using key parts of the input.

        * **3. Tokenization (Breaking Down Words):**
            - Converts words into numerical tokens for processing. Example: “summarize” → token 123.

        * **4. Fine-Tuning:**
            - **Base FLAN-T5:** Pretrained on general language tasks.
            - **Full Fine-Tuning:** Trains the whole model on summaries.
            - **PEFT with LoRA:** Adds small trainable layers for faster, resource-efficient tuning.

        * **5. Inference & Decoding:**
            - Encoder processes the input.
            - Decoder generates the summary token-by-token.
            - Decoded output becomes readable text.

        🚀 This pipeline enables deep understanding and concise summarization of dialogue!
        """
    )


st.markdown("---")

# --- ROUGE Score Visualizations ---
st.header("📊 Model Performance: ROUGE Score Analysis")
st.markdown(
    """
    To understand how well our models perform, we use **ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores**.
    It measures the overlap of words and phrases. Higher ROUGE scores indicate better summarization quality.

    * **ROUGE-1 (Unigram Overlap)**: Measures the overlap of individual words between the generated and reference summaries.
    * **ROUGE-2 (Bigram Overlap)**: Measures the overlap of pairs of words (bigrams).
    * **ROUGE-L (Longest Common Subsequence)**: Measures the longest sequence of words that appear in both summaries, regardless of their order.

    The plots below visually demonstrate the performance improvements achieved through fine-tuning,
    showing how the specialized training helps the models generate more accurate and relevant summaries compared to the base model.
    """
)

# Display ROUGE comparison plot
st.subheader("ROUGE Score Comparison Across Models")
rouge_comparison_path = os.path.join("assets", "rouge_comparison_plot.png")
if os.path.exists(rouge_comparison_path):
    st.image(rouge_comparison_path, caption="Comparison of ROUGE Scores (F1-Score) for Different FLAN-T5 Models", use_container_width=True)
else:
    st.warning(f"ROUGE Score Comparison Plot not found at: {rouge_comparison_path}")
    st.markdown("Please ensure `rouge_comparison_plot.png` is in the `assets` directory.")

st.markdown("---")

# Display ROUGE improvement comparison plot
st.subheader("ROUGE Score Improvement from Fine-Tuning")
rouge_improvement_path = os.path.join("assets", "rouge_improvement_comparison.png")
if os.path.exists(rouge_improvement_path):
    st.image(rouge_improvement_path, caption="ROUGE Score Improvement of Fine-Tuned Models over Base Model", use_container_width=True)
else:
    st.warning(f"ROUGE Score Improvement Comparison Plot not found at: {rouge_improvement_path}")
    st.markdown("Please ensure `rouge_improvement_comparison.png` is in the `assets` directory.")

st.markdown("---")
st.markdown("<center><sub>Designed and Developed by Nabeel Shan — Practical Applications of LLMs in Action</sub></center>", unsafe_allow_html=True)