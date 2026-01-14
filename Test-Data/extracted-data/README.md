# Extracted Test Data

`Test-Data/extracted-data` contains numbered case folders (`001`–`033`). Each case folder has the same three files:
- `query.txt`: the raw clinical query.
- `ground_truth.txt`: the expected answer for the case.
- `meta_eval.json`: metadata combining the query, ground truth answer, and disease label.

Folder layout:
- `001/`
- `002/`
- `003/`
- `…`
- `033/`

Example contents for case `001`:
- `query.txt`  
  ```
  A 53-year-old woman presented with fever, cough, and malaise after returning from a visit to Lahore. On examination, her temperature was 38°C and she had a rash on her upper chest. A chest X-ray showed patchy basal consolidation and a full blood count revealed a relative lymphocytosis. Malaria films were negative. Blood cultures were drawn and later grew gram-negative bacilli.
  ```
- `ground_truth.txt`  
  ```
  The diagnosis is enteric fever caused by Salmonella typhi. This is supported by the patient's presentation with a febrile illness, rash, and relative lymphocytosis after returning from an endemic area (Lahore). The diagnosis was confirmed by the isolation of Salmonella typhi from blood cultures.
  ```
- `meta_eval.json`  
  ```json
  {
    "test_id": "1",
    "query_for_retriever": "A 53-year-old woman presented with fever, cough, and malaise after returning from a visit to Lahore. On examination, her temperature was 38°C and she had a rash on her upper chest. A chest X-ray showed patchy basal consolidation and a full blood count revealed a relative lymphocytosis. Malaria films were negative. Blood cultures were drawn and later grew gram-negative bacilli.",
    "ground_truth_answer": "The diagnosis is enteric fever caused by Salmonella typhi. This is supported by the patient's presentation with a febrile illness, rash, and relative lymphocytosis after returning from an endemic area (Lahore). The diagnosis was confirmed by the isolation of Salmonella typhi from blood cultures.",
    "ground_truth_disease": "Enteric (typhoid) fever"
  }
  ```
