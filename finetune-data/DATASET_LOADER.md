# Dataset Loader for Qwen2-VL-2B Fine-tuning

This document provides instructions for loading the `train_structured.jsonl` dataset for fine-tuning **Qwen2-VL-2B** using Qwen's official scripts and tools.

---

## 📂 Dataset Structure

```
finetune-data/
├── train_structured.jsonl    # Training data in JSONL format
├── images/                   # Image files referenced in the dataset
│   ├── case_01_01.jpg
│   ├── case_01_02.jpg
│   └── ...
└── DATASET_LOADER.md         # This file
```

### Data Format

Each line in `train_structured.jsonl` is a JSON object following Qwen2-VL's conversation format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "images/case_01_01.jpg"},
        {"type": "image", "image": "images/case_01_02.jpg"},
        {"type": "text", "text": "Clinical case 01.\nSymptoms: ..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Symptoms: ...\n\nDiagnosis: ...\n\nConfirmation/Key evidence: ..."}
      ]
    }
  ],
  "meta": {
    "case_id": "01",
    "used_images": ["images/case_01_01.jpg", "images/case_01_02.jpg"],
    "symptoms": "...",
    "diagnosis": "...",
    "confirmation": "..."
  }
}
```

---

## 🚀 Installation

### Prerequisites

```bash
# Install Qwen2-VL dependencies
pip install transformers>=4.37.0
pip install torch>=2.0.0
pip install Pillow
pip install accelerate
pip install peft  # For LoRA fine-tuning

# Install Qwen's official packages
pip install qwen-vl-utils
```

---

## 📥 Dataset Loading Methods

### Method 1: Using Qwen's Native Data Processing

```python
import json
from pathlib import Path
from PIL import Image
from qwen_vl_utils import process_vision_info

def load_qwen2vl_dataset(jsonl_path: str, base_dir: str = None):
    """
    Load dataset in Qwen2-VL format.
    
    Args:
        jsonl_path: Path to the train_structured.jsonl file
        base_dir: Base directory for resolving image paths (defaults to jsonl parent dir)
    
    Returns:
        List of conversation samples
    """
    if base_dir is None:
        base_dir = Path(jsonl_path).parent
    else:
        base_dir = Path(base_dir)
    
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Resolve image paths to absolute paths
                messages = data['messages']
                for message in messages:
                    if isinstance(message['content'], list):
                        for item in message['content']:
                            if item.get('type') == 'image':
                                item['image'] = str(base_dir / item['image'])
                samples.append({
                    'messages': messages,
                    'meta': data.get('meta', {})
                })
    return samples

# Usage
dataset = load_qwen2vl_dataset('train_structured.jsonl')
print(f"Loaded {len(dataset)} samples")
```

### Method 2: Hugging Face Dataset Format

```python
import json
from datasets import Dataset
from pathlib import Path

def load_as_hf_dataset(jsonl_path: str, base_dir: str = None):
    """
    Load dataset as Hugging Face Dataset object.
    """
    if base_dir is None:
        base_dir = Path(jsonl_path).parent
    else:
        base_dir = Path(base_dir)
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # Process messages to resolve image paths
                messages = sample['messages']
                for message in messages:
                    if isinstance(message['content'], list):
                        for item in message['content']:
                            if item.get('type') == 'image':
                                item['image'] = str(base_dir / item['image'])
                
                data.append({
                    'messages': json.dumps(messages),  # Serialize for HF Dataset
                    'case_id': sample.get('meta', {}).get('case_id', ''),
                    'diagnosis': sample.get('meta', {}).get('diagnosis', ''),
                })
    
    return Dataset.from_list(data)

# Usage
hf_dataset = load_as_hf_dataset('train_structured.jsonl')
print(hf_dataset)
```

---

## 🔧 Full Training Script with Qwen2-VL

### Using Transformers + Qwen Scripts

```python
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from qwen_vl_utils import process_vision_info

# ==========================================
# Configuration
# ==========================================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATASET_PATH = "finetune-data/train_structured.jsonl"
OUTPUT_DIR = "./qwen2vl-finetuned"

# ==========================================
# Load Model and Processor
# ==========================================
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# ==========================================
# Dataset Class
# ==========================================
class Qwen2VLDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.base_dir = Path(jsonl_path).parent
        
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        
        # Build conversation with resolved image paths
        processed_messages = []
        for msg in messages:
            new_msg = {"role": msg["role"], "content": []}
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "image":
                        img_path = self.base_dir / item["image"]
                        new_msg["content"].append({
                            "type": "image",
                            "image": str(img_path)
                        })
                    else:
                        new_msg["content"].append(item)
            else:
                new_msg["content"] = msg["content"]
            processed_messages.append(new_msg)
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Process images
        image_inputs, video_inputs = process_vision_info(processed_messages)
        
        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Create labels (shift input_ids for causal LM)
        inputs["labels"] = inputs["input_ids"].clone()
        
        return {k: v.squeeze(0) for k, v in inputs.items()}

# ==========================================
# Data Collator
# ==========================================
def collate_fn(batch):
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100
    )
    
    # Handle pixel values (images)
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

# ==========================================
# Initialize Dataset and Training
# ==========================================
train_dataset = Qwen2VLDataset(DATASET_PATH, processor)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

# Start training
trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
```

---

## 🎯 LoRA Fine-tuning (Recommended for Limited Resources)

```python
import json
import torch
from pathlib import Path
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==========================================
# Model Configuration
# ==========================================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATASET_PATH = "finetune-data/train_structured.jsonl"
OUTPUT_DIR = "./qwen2vl-lora-finetuned"

# Load model in 4-bit for efficient training
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# ==========================================
# LoRA Configuration
# ==========================================
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# Load Dataset Function
# ==========================================
def load_dataset_for_sft(jsonl_path: str):
    """Format dataset for SFTTrainer."""
    base_dir = Path(jsonl_path).parent
    formatted_data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                messages = sample['messages']
                
                # Resolve image paths
                for msg in messages:
                    if isinstance(msg['content'], list):
                        for item in msg['content']:
                            if item.get('type') == 'image':
                                item['image'] = str(base_dir / item['image'])
                
                formatted_data.append({'messages': messages})
    
    return formatted_data

# ==========================================
# Training
# ==========================================
train_data = load_dataset_for_sft(DATASET_PATH)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
)

# Note: For SFTTrainer with vision models, you may need custom data processing
# This is a simplified example - adjust based on your specific requirements
```

---

## 📋 Using LLaMA-Factory (Recommended)

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) provides excellent support for Qwen2-VL fine-tuning.

### 1. Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 2. Create Dataset Configuration

Create `data/dataset_info.json` entry:

```json
{
  "clinical_cases_qwen2vl": {
    "file_name": "path/to/train_structured.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

### 3. Training Configuration YAML

Create `examples/train_qwen2vl_clinical.yaml`:

```yaml
### Model
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 16
lora_dropout: 0.05

### Dataset
dataset: clinical_cases_qwen2vl
template: qwen2_vl
cutoff_len: 2048
preprocessing_num_workers: 16

### Output
output_dir: saves/qwen2vl-clinical
logging_steps: 10
save_steps: 500
plot_loss: true

### Training Hyperparameters
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### Evaluation
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### 4. Run Training

```bash
llamafactory-cli train examples/train_qwen2vl_clinical.yaml
```

---

## ✅ Validation Script

```python
import json
from pathlib import Path
from PIL import Image

def validate_dataset(jsonl_path: str):
    """Validate dataset integrity."""
    base_dir = Path(jsonl_path).parent
    issues = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            # Check required fields
            if 'messages' not in data:
                issues.append(f"Line {line_num}: Missing 'messages' field")
                continue
            
            # Validate images exist
            for msg in data['messages']:
                if isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if item.get('type') == 'image':
                            img_path = base_dir / item['image']
                            if not img_path.exists():
                                issues.append(f"Line {line_num}: Missing image {item['image']}")
                            else:
                                # Validate image can be opened
                                try:
                                    Image.open(img_path).verify()
                                except Exception as e:
                                    issues.append(f"Line {line_num}: Invalid image {item['image']} - {e}")
    
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("✅ Dataset validation passed!")
    
    return len(issues) == 0

# Run validation
validate_dataset('train_structured.jsonl')
```

---

## 📊 Dataset Statistics

```python
import json
from collections import Counter

def get_dataset_stats(jsonl_path: str):
    """Get dataset statistics."""
    stats = {
        'total_samples': 0,
        'total_images': 0,
        'images_per_sample': [],
        'diagnoses': Counter(),
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                stats['total_samples'] += 1
                
                # Count images
                image_count = 0
                for msg in data['messages']:
                    if isinstance(msg.get('content'), list):
                        for item in msg['content']:
                            if item.get('type') == 'image':
                                image_count += 1
                
                stats['total_images'] += image_count
                stats['images_per_sample'].append(image_count)
                
                # Track diagnoses
                if 'meta' in data and 'diagnosis' in data['meta']:
                    stats['diagnoses'][data['meta']['diagnosis']] += 1
    
    # Calculate averages
    avg_images = sum(stats['images_per_sample']) / len(stats['images_per_sample'])
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Average images per sample: {avg_images:.2f}")
    print(f"   Unique diagnoses: {len(stats['diagnoses'])}")
    print(f"\n   Top 10 diagnoses:")
    for diagnosis, count in stats['diagnoses'].most_common(10):
        print(f"     - {diagnosis}: {count}")
    
    return stats

get_dataset_stats('train_structured.jsonl')
```

---

## 🔗 References

- [Qwen2-VL Official Repository](https://github.com/QwenLM/Qwen2-VL)
- [Qwen2-VL Model Card (Hugging Face)](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen VL Utils](https://github.com/QwenLM/Qwen-VL)

---

## ⚠️ Notes

1. **Image Paths**: All image paths in the dataset are relative to the `finetune-data/` directory
2. **Multi-image Support**: Each sample can contain 1-6 images
3. **Task Type**: Medical diagnosis from clinical images and symptoms
4. **Format Compatibility**: The format is compatible with Qwen2-VL's expected conversation structure
