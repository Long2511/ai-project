import pandas as pd
import re
from collections import Counter

# 1. Define Core Logic
def normalize(t):
    """Clean text by removing markdown, punctuation, and lowering case."""
    return re.sub(r'[^\w\s]', '', str(t).replace("**", "").lower().strip())

def get_f1(p, g):
    """Calculate token-level overlap F1 score (SQuAD Standard)."""
    p_t, g_t = normalize(p).split(), normalize(g).split()
    if not p_t or not g_t: return 0.0
    common = Counter(p_t) & Counter(g_t)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    prec, rec = num_same/len(p_t), num_same/len(g_t)
    return (2 * prec * rec) / (prec + rec)


# If the F1 score is above this threshold, consider it a partial match
PARTIAL_MATCH_F1_CRITERIA = 0.4

def categorize(p, g):
    """Determine if a match is Exact, Partial (clinical subtype), or Incorrect."""
    norm_p, norm_g = normalize(p), normalize(g)
    f1 = get_f1(p, g)
    if norm_p == norm_g: return "Exact Match"
    # Logic for almost matches: Substring or high token overlap
    if norm_p in norm_g or norm_g in norm_p or f1 > PARTIAL_MATCH_F1_CRITERIA: 
        return "Partial (Clinical Subtype)"
    return "Incorrect"

# 2. Load Data
df = pd.read_excel('test_dataset_extracted_answer_disease.xlsx', sheet_name='Sheet1')
models = ['gwen2_base', 'gwen2_finetune', 'gwen2_rag', 'gwen2_finetune_rag', 'gemini_rag']

# 3. Process Detailed Results (Sheet 1 Content)
detailed_rows = []
for idx, row in df.iterrows():
    entry = {"ID": row['test_id'], "Ground Truth": row['ground_truth']}
    for m in models:
        pred = row[m]
        status = categorize(pred, row['ground_truth'])
        entry[f"{m}_Answer"] = pred
        entry[f"{m}_Status"] = status
    detailed_rows.append(entry)

appendix_df = pd.DataFrame(detailed_rows)

# 4. Generate Summary Table (Sheet 2 Content)
summary = []
for m in models:
    counts = appendix_df[f"{m}_Status"].value_counts(normalize=True) * 100
    summary.append({
        "Model": m,
        "Exact Match (%)": round(counts.get("Exact Match", 0), 2),
        "Partial Match (%)": round(counts.get("Partial (Clinical Subtype)", 0), 2),
        "Total Success (%)": round(counts.get("Exact Match", 0) + counts.get("Partial (Clinical Subtype)", 0), 2)
    })
summary_df = pd.DataFrame(summary)

# 5. Save to Excel with Multiple Sheets
with pd.ExcelWriter('automatic_metric_result.xlsx', engine='openpyxl') as writer:
    appendix_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print("Evaluation complete. Results saved to 'automatic_metric_result.xlsx'.")