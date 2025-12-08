import streamlit as st
import pandas as pd
import os
from rubrics import RUBRICS

st.set_page_config(layout="wide", page_title="LLM QA Annotator")

# --- Constants & Config ---
EXCLUDED_COLS = [
    'chunk_id', 'agency', 'section', 'text_chunk', 
    'question', 'answer', 'useful_text_chunk'
]

# --- Helper Functions ---
def get_model_columns(df):
    """Identify columns that contain generated answers."""
    all_cols = df.columns.tolist()
    # Filter out metadata columns and existing score columns
    model_cols = [
        c for c in all_cols 
        if c not in EXCLUDED_COLS 
        and not c.lower().strip().endswith('score')
        and not c.startswith('Unnamed')
    ]
    return model_cols

def load_data(uploaded_file):
    try:
        # Try reading with default settings
        df = pd.read_csv(uploaded_file)
    except Exception:
        # Fallback for bad lines or encoding issues if necessary
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'selected_rubrics' not in st.session_state:
    st.session_state.selected_rubrics = list(RUBRICS.keys())
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# --- Sidebar: Configuration ---
st.sidebar.title("Settings")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # Load data only if it's a new file or state is empty
    if st.session_state.data is None:
        df = load_data(uploaded_file)
        st.session_state.data = df
        st.session_state.current_index = 0
        st.success("File loaded!")

# 2. Rubric Selection
st.sidebar.subheader("Select Rubrics to Grade")
all_rubrics = list(RUBRICS.keys())
selected = st.sidebar.multiselect(
    "Choose rubrics:", 
    all_rubrics,
    default=all_rubrics
)
st.session_state.selected_rubrics = selected

# 3. Progress & Navigation
if st.session_state.data is not None:
    df = st.session_state.data
    total_rows = len(df)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    
    # Jump to index
    idx = st.sidebar.number_input(
        "Go to Row Index", 
        min_value=0,
        max_value=total_rows-1,
        value=st.session_state.current_index
    )
    if idx != st.session_state.current_index:
        st.session_state.current_index = idx
        st.rerun()

    # Progress Bar
    progress = (st.session_state.current_index + 1) / total_rows
    st.sidebar.progress(progress)
    st.sidebar.text(f"Row {st.session_state.current_index + 1} of {total_rows}")

    # Save Progress
    st.sidebar.markdown("---")
    st.sidebar.subheader("Save Progress")
    csv_data = convert_df_to_csv(df)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="annotated_progress.csv",
        mime="text/csv",
    )

# --- Main Interface ---
if st.session_state.data is not None:
    df = st.session_state.data
    curr_idx = st.session_state.current_index
    row = df.iloc[curr_idx]

    # Identify Model Columns
    model_cols = get_model_columns(df)
    
    if not model_cols:
        st.error("No generated answer columns found based on exclusion criteria.")
    else:
        # Model Selector
        st.header("Annotation Workspace")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            target_model = st.selectbox("Select Model to Evaluate", model_cols)
        
        # --- Layout: Context & Comparison ---
        with st.expander("Show Context & Ground Truth", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Context (Text Chunk)")
                st.info(row.get('text_chunk', 'N/A'))
                st.markdown("### Question")
                st.warning(row.get('question', 'N/A'))
            with c2:
                st.markdown("### Ground Truth Answer")
                st.success(row.get('answer', 'N/A'))

        st.markdown("---")
        
        # Display Model Answer
        st.markdown(f"### Generated Answer: `{target_model}`")
        model_answer = row.get(target_model, "")
        
        # Handle potential float/nan values in answer
        if pd.isna(model_answer):
            model_answer = "(Empty/NaN)"
        else:
            model_answer = str(model_answer)
            
        st.markdown(
            f"""
            <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
                {model_answer}
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.subheader("Rubric Scoring")

        # Form for scoring
        with st.form(key=f"scoring_form_{curr_idx}_{target_model}"):
            
            # Create columns for rubrics to save space
            r_cols = st.columns(2)
            
            for i, rubric_name in enumerate(st.session_state.selected_rubrics):
                col_idx = i % 2
                with r_cols[col_idx]:
                    rubric_data = RUBRICS[rubric_name]
                    
                    # Determine column name for this score
                    # Convention: {ModelName}_{RubricName}_score
                    score_col_name = f"{target_model}_{rubric_name}_score"
                    
                    # Get existing value if any
                    current_val = 0
                    if score_col_name in df.columns:
                        val = df.at[curr_idx, score_col_name]
                        if not pd.isna(val):
                            current_val = int(val)
                    
                    st.markdown(f"**{rubric_name}**")
                    
                    if rubric_name == "Generation Quality":
                        options = [0, 1]
                        default_index = options.index(current_val) if current_val in options else 0
                    else:
                        options = [-2, -1, 0, 1, 2]
                        default_index = options.index(current_val) if current_val in options else 2

                    score = st.radio(
                        label=f"Score for {rubric_name}",
                        options=options,
                        index=default_index,
                        format_func=lambda x: f"{x} : {rubric_data[x]}",
                        key=f"{rubric_name}_{curr_idx}",
                        label_visibility="collapsed"
                    )

            st.markdown("---")
            
            # Navigation Buttons inside form to trigger submit
            b_col1, b_col2, b_col3 = st.columns([1, 1, 4])
            
            with b_col1:
                prev_clicked = st.form_submit_button("Previous")
            with b_col2:
                next_clicked = st.form_submit_button("Save & Next")
                
            if prev_clicked:
                # Save current state before moving?
                # Since sliders update state on interaction, we just need to write to DF
                for rubric_name in st.session_state.selected_rubrics:
                    score_col = f"{target_model}_{rubric_name}_score"
                    # Get value from session state key
                    val = st.session_state[f"{rubric_name}_{curr_idx}"]
                    st.session_state.data.at[curr_idx, score_col] = val
                
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()

            if next_clicked:
                # Save data
                for rubric_name in st.session_state.selected_rubrics:
                    score_col = f"{target_model}_{rubric_name}_score"
                    val = st.session_state[f"{rubric_name}_{curr_idx}"]
                    st.session_state.data.at[curr_idx, score_col] = val
                
                if st.session_state.current_index < total_rows - 1:
                    st.session_state.current_index += 1
                    st.rerun()
                else:
                    st.success("You have reached the end of the file.")

else:
    st.info("Please upload a CSV file to start.")
