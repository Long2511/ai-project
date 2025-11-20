import pandas as pd
import ast
import os
import re
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)

# Load Data
DATASET_FILE = "testcase_medgemma_with_rag.xlsx"
FILE_NO_RAG = "testcase_medgemma_without_rag.xlsx"


# --- 1. Configuration & Setup ---
# Initialize Llama3 via Ollama
langchain_llm = ChatOllama(
    model="llama3:instruct",
    temperature=0.0,     # Absolute zero for stability
    format="json",
    num_predict=4096,    # Ensure enough context window for reasoning
    num_ctx=8192,
    keep_alive="60m"     # Keep model loaded for speed
)

run_config = RunConfig(timeout=600, max_retries=10, max_workers=1)


langchain_embeddings = OllamaEmbeddings(model="llama3:instruct")

# --- 2. Helper Function to Parse Data ---
def parse_dataset(file_path, has_context=True, sheet_name='Sheet1'):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        print(f"   Reading {file_path}...")
        df = pd.read_excel(file_path, sheet_name=None)
        if sheet_name in df:
            df = df[sheet_name]
        else:
            first_sheet = list(df.keys())[0]
            print(f"Sheet '{sheet_name}' not found. Using first sheet: '{first_sheet}'")
            df = df[first_sheet]
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

    data_entries = []

    for index, row in df.iterrows():
        try:
            # --- EXTRACT QUESTION ---
            raw_history = str(row.get('answer', '[]')) 
            question = ""
            
            # Strategy 1: Try strict parsing (Best Quality)
            try:
                if not pd.isna(raw_history):
                    history = ast.literal_eval(raw_history)
                    for turn in history:
                        if turn.get('role') == 'user':
                            content = turn.get('content')
                            if isinstance(content, list):
                                for item in content:
                                    if item.get('type') == 'text':
                                        question = item.get('text', '')
                                        break
                            elif isinstance(content, str):
                                question = content
                            break
            except Exception:
                # Strategy 2: Regex Fallback
                try:
                    # Look for text inside the user role
                    match = re.search(r"'role':\s*'user'.*?'text':\s*['\"](.*?)['\"]", raw_history, re.DOTALL)
                    if match:
                        question = match.group(1)
                except Exception:
                    pass

            # --- CLEANING THE QUESTION ---
            if question:
                # 1. Replace escaped newlines
                question = question.replace('\\n', '\n')
                
                # 2. Split by "Question:" if present and take the second part
                if "Question:" in question:
                    parts = question.split("Question:", 1)
                    if len(parts) > 1:
                        question = parts[1]
                
                # 3. Remove the specific preamble if "Question:" wasn't there
                preamble = "You are diagnosing tropical diseases for a patient:"
                question = question.replace(preamble, "")
                
                # 4. Final strip of whitespace
                question = question.strip()

            if not question:
                question = "Could not extract question"
                print(f"Row {index}: Extraction failed or empty.")

            # --- EXTRACT ANSWER ---
            ragas_answer = str(row.get('model_response', ''))
            if pd.isna(ragas_answer): ragas_answer = ""

            # --- EXTRACT GROUND TRUTH ---
            ground_truth = str(row.get('ground_truth', ''))
            if pd.isna(ground_truth): ground_truth = ""
            
            case_id = str(row.get('case_id', index))

            entry = {
                "case_id": case_id,
                "question": question,
                "answer": ragas_answer,
                "ground_truth": ground_truth
            }

            # --- EXTRACT CONTEXTS ---
            if has_context:
                raw_context = row.get('context', '')
                if pd.isna(raw_context):
                    entry["contexts"] = [""]
                else:
                    entry["contexts"] = [str(raw_context)]
            
            data_entries.append(entry)
            
        except Exception as e:
            print(f"Skipping row {index} due to error: {e}")
            continue

    return Dataset.from_list(data_entries)



def save_evaluation_result(rag_results):
    # Evaluate RAG
    print("\n=== Evaluating RAG Pipeline ===")

    df_rag_results = rag_results.to_pandas()
    # Add Case ID back for mapping
    df_rag_results['case_id'] = dataset_rag['case_id']

    # --- CHECKPOINT SAVE ---
    checkpoint_rag = "intermediate_rag_results.xlsx"
    print(f"Saving RAG checkpoint to {checkpoint_rag}...")
    df_rag_results.to_excel(checkpoint_rag, index=False)
    print("RAG Checkpoint saved.")


    # --- 5. Comparison and Saving ---

    print("\nPreparing Final Report...")

    # 1. Convert original Datasets to Pandas so we have the source columns (question, answer, etc.)
    df_rag_source = dataset_rag.to_pandas()

    # 2. Merge Scores into Source Data for RAG
    # This ensures 'question', 'ground_truth', etc. exist alongside 'answer_correctness'
    if 'case_id' not in df_rag_results.columns and len(df_rag_results) == len(df_rag_source):
        df_rag_results['case_id'] = df_rag_source['case_id']

    df_rag_full = pd.merge(df_rag_source, df_rag_results, on='case_id', how='left')


    # 4. Define columns dynamically (in case a metric failed completely)
    rag_cols = ['case_id', 'question', 'ground_truth', 'answer', 'contexts']
    for m in ['answer_correctness', 'faithfulness', 'context_recall', 'context_precision']:
        if m in df_rag_full.columns:
            rag_cols.append(m)
            


    # 5. Final Merge: RAG vs No-RAG
    comparison_df = pd.merge(
        df_rag_full[rag_cols],
        on='case_id',
        suffixes=('_rag', '_no_rag'),
        how='inner'
    )

    # 6. Calculate Improvement
    if 'answer_correctness_rag' in comparison_df.columns and 'answer_correctness_no_rag' in comparison_df.columns:
        comparison_df['correctness_improvement'] = comparison_df['answer_correctness_rag'] - comparison_df['answer_correctness_no_rag']

    # Save Results
    output_file = "RAG_Evaluation_Report.xlsx"
    try:
        with pd.ExcelWriter(output_file) as writer:
            comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
            
            # Calculate averages
            avg_scores = comparison_df.mean(numeric_only=True)
            summary_data = {
                'Metric': avg_scores.index.tolist(),
                'Score': avg_scores.values.tolist()
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary_Stats', index=False)
        
        print(f"\nEvaluation complete. Results saved to {output_file}")
        print("Summary:")
        print(pd.DataFrame(summary_data))
        
    except Exception as e:
        print(f"Error saving Excel file: {e}")




# --- 1. Load and Prepare Datasets ---
dataset_rag = parse_dataset(DATASET_FILE)

# --- 2. Evaluation Execution ---
# Metrics for RAG (Retrieval + Generation)
metrics_rag = [
    answer_correctness,
    faithfulness,
    context_recall,
    context_precision
]

rag_results = evaluate(
    dataset_rag,
    metrics=metrics_rag,
    llm=langchain_llm,
    embeddings=langchain_embeddings,
    run_config=run_config
)

# --- 3. Save Result to file ---
save_evaluation_result(rag_results)


