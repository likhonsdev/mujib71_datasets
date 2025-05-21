import os
import sys
import json
import pandas as pd
import shutil
from huggingface_hub import HfApi, create_repo, upload_folder

def upload_dataset_to_huggingface():
    """
    Upload the dataset folder to Hugging Face Hub
    """
    print("Starting upload process to Hugging Face Hub...")
    
    # Get repository information
    repo_id = "likhonsheikhdev/mujib71_datasets"  # Main repository
    task_specific_repos = {
        "chat": "likhonsheikhdev/mujib71_chat",
        "classify_multi": "likhonsheikhdev/mujib71_classify_multi",
        "classify_single": "likhonsheikhdev/mujib71_classify_single",
        "rerank": "likhonsheikhdev/mujib71_rerank"
    }
    
    # Initialize Hugging Face API (will use token stored during huggingface-cli login)
    api = HfApi()
    
    try:
        # Check if dataset exists in the current directory
        if not os.path.exists("dataset"):
            print("Error: 'dataset' directory not found. Please run the scraper.py script first.")
            return False
            
        # Create task-specific datasets directory
        os.makedirs("dataset/task_specific", exist_ok=True)
        
        # Check for and create task-specific datasets
        # 1. Chat dataset
        if not os.path.exists("dataset/task_specific/chat"):
            print("Creating Chat dataset...")
            import create_chat_dataset
            create_chat_dataset.create_chat_dataset()
            
        # 2. Multi-label classification dataset
        if not os.path.exists("dataset/task_specific/classify_multi"):
            print("Creating Multi-label Classification dataset...")
            create_multilabel_dataset()
            
        # 3. Single-label classification dataset
        if not os.path.exists("dataset/task_specific/classify_single"):
            print("Creating Single-label Classification dataset...")
            create_singlelabel_dataset()
            
        # 4. Rerank dataset
        if not os.path.exists("dataset/task_specific/rerank"):
            print("Creating Rerank dataset...")
            create_rerank_dataset()
        
        # Create main repository if it doesn't exist
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            print(f"Main repository {repo_id} already exists.")
        except Exception:
            print(f"Creating main dataset repository: {repo_id}")
            create_repo(repo_id=repo_id, repo_type="dataset")
        
        # Create task-specific repositories if they don't exist
        for task, task_repo in task_specific_repos.items():
            try:
                api.repo_info(repo_id=task_repo, repo_type="dataset")
                print(f"Repository {task_repo} already exists.")
            except Exception:
                print(f"Creating task-specific repository for {task}: {task_repo}")
                create_repo(repo_id=task_repo, repo_type="dataset")
        
        # Upload the main dataset folder
        print(f"Uploading main dataset to {repo_id}...")
        api.upload_folder(
            folder_path="dataset",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload Sheikh Mujibur Rahman Bangla NLP Dataset"
        )
        
        # Upload task-specific datasets
        for task, task_repo in task_specific_repos.items():
            task_folder = f"dataset/task_specific/{task}"
            if os.path.exists(task_folder):
                print(f"Uploading {task} dataset to {task_repo}...")
                api.upload_folder(
                    folder_path=task_folder,
                    repo_id=task_repo,
                    repo_type="dataset",
                    commit_message=f"Upload Sheikh Mujibur Rahman {task.replace('_', ' ').title()} Dataset"
                )
        
        # Upload README.md to all repositories
        if os.path.exists("README.md"):
            print("Uploading README.md to main repository...")
            api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add dataset documentation"
            )
            
            # Create and upload task-specific READMEs
            create_and_upload_task_readme(api, "chat", task_specific_repos["chat"])
            create_and_upload_task_readme(api, "classify_multi", task_specific_repos["classify_multi"])
            create_and_upload_task_readme(api, "classify_single", task_specific_repos["classify_single"])
            create_and_upload_task_readme(api, "rerank", task_specific_repos["rerank"])
            
        print("\n✅ Successfully uploaded datasets to Hugging Face Hub")
        print(f"Main Dataset: https://huggingface.co/datasets/{repo_id}")
        for task, task_repo in task_specific_repos.items():
            print(f"{task.replace('_', ' ').title()} Dataset: https://huggingface.co/datasets/{task_repo}")
        return True
        
    except Exception as e:
        print(f"Error uploading dataset: {str(e)}")
        return False

def create_multilabel_dataset():
    """Create a multi-label classification dataset from the existing data"""
    os.makedirs("dataset/task_specific/classify_multi", exist_ok=True)
    
    # Load the original data
    with open("dataset/mujib71_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Define categories for multi-label classification
    categories = ["politics", "independence", "family", "legacy", "governance", "speeches", "international"]
    
    # Process news articles
    classification_examples = []
    for article in data.get("news_articles", []):
        title = article.get("title", "")
        content = article.get("content", "")
        
        if not title or not content:
            continue
        
        # Determine labels based on content keywords (simplified approach)
        labels = []
        if any(kw in content.lower() for kw in ["political", "politics", "government", "minister", "president"]):
            labels.append("politics")
            
        if any(kw in content.lower() for kw in ["independence", "1971", "liberation", "war", "freedom"]):
            labels.append("independence")
            
        if any(kw in content.lower() for kw in ["family", "daughter", "son", "wife", "children"]):
            labels.append("family")
            
        if any(kw in content.lower() for kw in ["legacy", "influence", "remembrance", "memory", "commemoration"]):
            labels.append("legacy")
            
        if any(kw in content.lower() for kw in ["governance", "policy", "administration", "leadership", "development"]):
            labels.append("governance")
            
        if any(kw in content.lower() for kw in ["speech", "address", "declaration", "statement", "message"]):
            labels.append("speeches")
            
        if any(kw in content.lower() for kw in ["international", "foreign", "relations", "diplomacy", "visit"]):
            labels.append("international")
        
        # Ensure at least one label
        if not labels:
            labels = ["politics"]  # Default category
        
        # Create classification example
        example = {
            "text": content,
            "title": title,
            "labels": labels
        }
        classification_examples.append(example)
    
    # Split into train and test
    train_size = int(len(classification_examples) * 0.8)
    train_examples = classification_examples[:train_size]
    test_examples = classification_examples[train_size:]
    
    # Save as JSONL files
    with open("dataset/task_specific/classify_multi/train.jsonl", "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    with open("dataset/task_specific/classify_multi/test.jsonl", "w", encoding="utf-8") as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Create metadata with label information
    metadata = {
        "labels": categories
    }
    
    with open("dataset/task_specific/classify_multi/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Created multi-label classification dataset with {len(train_examples)} training and {len(test_examples)} test examples")
    return True

def create_singlelabel_dataset():
    """Create a single-label classification dataset from the existing data"""
    os.makedirs("dataset/task_specific/classify_single", exist_ok=True)
    
    # Load the original data
    with open("dataset/mujib71_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Define categories for single-label classification (time periods)
    categories = ["pre-independence", "independence-war", "post-independence", "legacy"]
    
    # Process news articles and Wikipedia content
    classification_examples = []
    
    # Process news articles
    for article in data.get("news_articles", []):
        title = article.get("title", "")
        content = article.get("content", "")
        
        if not title or not content:
            continue
        
        # Determine primary label based on content (simplified approach)
        label = None
        
        # Check for pre-independence content
        if any(kw in content.lower() for kw in ["before 1971", "pre-independence", "six points", "pakistan era"]):
            label = "pre-independence"
        # Check for independence war content
        elif any(kw in content.lower() for kw in ["liberation war", "1971 war", "independence struggle", "march 1971"]):
            label = "independence-war"
        # Check for post-independence content
        elif any(kw in content.lower() for kw in ["after independence", "1972", "1973", "1974", "1975", "constitution"]):
            label = "post-independence"
        # Check for legacy content
        elif any(kw in content.lower() for kw in ["legacy", "remembering", "memory", "commemoration", "influence"]):
            label = "legacy"
        else:
            # If no clear match, try to infer from title
            if "legacy" in title.lower() or "remembering" in title.lower():
                label = "legacy"
            elif "independence" in title.lower() or "1971" in title.lower():
                label = "independence-war"
            elif "1972" in title.lower() or "constitution" in title.lower():
                label = "post-independence"
            else:
                # Default label if can't determine
                continue
        
        # Create classification example
        example = {
            "text": content,
            "title": title,
            "label": label
        }
        classification_examples.append(example)
    
    # Process Wikipedia sections
    for wiki_page in data.get("wikipedia", []):
        for section in wiki_page.get("sections", []):
            section_title = section.get("title", "")
            section_content = section.get("content", "")
            
            if not section_title or not section_content:
                continue
            
            # Determine label based on section title and content
            label = None
            
            if any(kw in section_title.lower() for kw in ["early life", "education", "political career", "pakistan"]) or \
               any(kw in section_content.lower() for kw in ["before independence", "six points", "pakistan era"]):
                label = "pre-independence"
            elif any(kw in section_title.lower() for kw in ["liberation", "1971", "independence war"]) or \
                 any(kw in section_content.lower() for kw in ["liberation war", "1971 war", "march 1971"]):
                label = "independence-war"
            elif any(kw in section_title.lower() for kw in ["after independence", "presidency", "prime minister"]) or \
                 any(kw in section_content.lower() for kw in ["after independence", "president", "prime minister", "constitution"]):
                label = "post-independence"
            elif any(kw in section_title.lower() for kw in ["legacy", "remembrance", "influence"]) or \
                 any(kw in section_content.lower() for kw in ["legacy", "remembering", "commemoration", "influence"]):
                label = "legacy"
            else:
                # Skip if can't determine
                continue
            
            # Create classification example
            example = {
                "text": section_content,
                "title": section_title,
                "label": label
            }
            classification_examples.append(example)
    
    # Split into train and test
    train_size = int(len(classification_examples) * 0.8)
    train_examples = classification_examples[:train_size]
    test_examples = classification_examples[train_size:]
    
    # Save as JSONL files
    with open("dataset/task_specific/classify_single/train.jsonl", "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    with open("dataset/task_specific/classify_single/test.jsonl", "w", encoding="utf-8") as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Create metadata with label information
    metadata = {
        "labels": categories
    }
    
    with open("dataset/task_specific/classify_single/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Created single-label classification dataset with {len(train_examples)} training and {len(test_examples)} test examples")
    return True

def create_rerank_dataset():
    """Create a rerank dataset from the existing data"""
    os.makedirs("dataset/task_specific/rerank", exist_ok=True)
    
    # Load the original data
    with open("dataset/mujib71_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process news articles and Wikipedia content
    rerank_examples = []
    
    # Create a list of all content paragraphs
    all_paragraphs = []
    
    # Add news article paragraphs
    for article in data.get("news_articles", []):
        content = article.get("content", "")
        if content:
            # Split content into paragraphs
            paragraphs = [p for p in content.split("\n\n") if len(p) > 100]  # Only use substantial paragraphs
            all_paragraphs.extend(paragraphs)
    
    # Add Wikipedia paragraphs
    for wiki_page in data.get("wikipedia", []):
        for section in wiki_page.get("sections", []):
            section_content = section.get("content", "")
            if section_content:
                # Split content into paragraphs
                paragraphs = [p for p in section_content.split("\n\n") if len(p) > 100]
                all_paragraphs.extend(paragraphs)
    
    # Create example queries
    queries = [
        "What was Sheikh Mujibur Rahman's role in the independence movement?",
        "How did Sheikh Mujibur Rahman contribute to Bangladesh's constitution?",
        "What were Sheikh Mujibur Rahman's major achievements?",
        "What happened to Sheikh Mujibur Rahman in 1971?",
        "What is Sheikh Mujibur Rahman's legacy?",
        "Who was Sheikh Mujibur Rahman?",
        "What was the Six-Point Movement?",
        "How did Bangladesh achieve independence?",
        "What was Sheikh Mujibur Rahman's foreign policy?",
        "What happened to Sheikh Mujibur Rahman's family?"
    ]
    
    # For each query, select relevant and less-relevant paragraphs
    import random
    
    for query in queries:
        # For each query, we need a set of documents with relevance scores
        documents = []
        
        # Select 5-10 random paragraphs
        selected_paragraphs = random.sample(all_paragraphs, min(10, len(all_paragraphs)))
        
        # Assign relevance scores (simulating real relevance)
        for i, paragraph in enumerate(selected_paragraphs):
            # Very basic relevance scoring based on keyword matching
            query_terms = query.lower().split()
            relevance = 0
            
            for term in query_terms:
                if term in paragraph.lower():
                    relevance += 1
            
            # Scale relevance to 0-5 range
            relevance = min(5, relevance)
            
            documents.append({
                "text": paragraph,
                "relevance": relevance
            })
        
        # Create rerank example
        example = {
            "query": query,
            "documents": documents
        }
        rerank_examples.append(example)
    
    # Split into train and test
    train_size = int(len(rerank_examples) * 0.7)
    train_examples = rerank_examples[:train_size]
    test_examples = rerank_examples[train_size:]
    
    # Save as JSONL files
    with open("dataset/task_specific/rerank/train.jsonl", "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    with open("dataset/task_specific/rerank/test.jsonl", "w", encoding="utf-8") as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Created rerank dataset with {len(train_examples)} training and {len(test_examples)} test examples")
    return True

def create_and_upload_task_readme(api, task, repo_id):
    """Create and upload a task-specific README.md file"""
    task_title = task.replace('_', ' ').title()
    
    # Task-specific descriptions
    task_descriptions = {
        "chat": """This dataset contains conversation pairs about Sheikh Mujibur Rahman (known as "Bangabandhu"), the founding father of Bangladesh, formatted for fine-tuning conversational AI models.""",
        
        "classify_multi": """This dataset contains multi-label classification examples about Sheikh Mujibur Rahman (known as "Bangabandhu"), the founding father of Bangladesh, with labels for different topics related to his life and legacy.""",
        
        "classify_single": """This dataset contains single-label classification examples about Sheikh Mujibur Rahman (known as "Bangabandhu"), the founding father of Bangladesh, categorized by the time periods of his life and legacy.""",
        
        "rerank": """This dataset contains examples for training reranking models to determine relevance of text passages to queries about Sheikh Mujibur Rahman (known as "Bangabandhu"), the founding father of Bangladesh."""
    }
    
    # Task-specific format explanations
    format_explanations = {
        "chat": """Each example is a JSON object with a "messages" array.
- Messages include roles: "System" (optional), "User", and "Chatbot"
- The dataset is split into training and validation sets""",
        
        "classify_multi": """Each example is a JSON object with:
- "text": The content to classify
- "title": The title of the content
- "labels": An array of labels from the set ["politics", "independence", "family", "legacy", "governance", "speeches", "international"]""",
        
        "classify_single": """Each example is a JSON object with:
- "text": The content to classify
- "title": The title of the content  
- "label": A single label from the set ["pre-independence", "independence-war", "post-independence", "legacy"]""",
        
        "rerank": """Each example is a JSON object with:
- "query": A question about Sheikh Mujibur Rahman
- "documents": An array of documents with "text" and "relevance" (0-5 scale)"""
    }
    
    # Task-specific usage examples
    usage_examples = {
        "chat": """```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("likhonsheikhdev/mujib71_chat")

# Access the splits
train_data = dataset["train"]
test_data = dataset["test"]

# Example usage with a chat model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Format a chat example
example = train_data[0]
prompt = f"<|system|>{example['messages'][0]['content']}<|endoftext|>\\n<|user|>{example['messages'][1]['content']}<|endoftext|>\\n<|assistant|>"
```""",
        
        "classify_multi": """```python
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the dataset
dataset = load_dataset("likhonsheikhdev/mujib71_classify_multi")

# Get labels from metadata
labels = dataset["metadata"]["labels"]

# Prepare tokenizer and model for multi-label classification
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", 
                                                         num_labels=len(labels),
                                                         problem_type="multi_label_classification")

# Process an example
example = dataset["train"][0]
inputs = tokenizer(example["text"], truncation=True, padding="max_length", return_tensors="pt")
```""",
        
        "classify_single": """```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load the dataset
dataset = load_dataset("likhonsheikhdev/mujib71_classify_single")

# Get labels from metadata
labels = dataset["metadata"]["labels"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# Prepare model for single-label classification
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)
```""",
        
        "rerank": """```python
from datasets import load_dataset
from sentence_transformers import CrossEncoder

# Load the dataset
dataset = load_dataset("likhonsheikhdev/mujib71_rerank")

# Prepare data for training a cross-encoder
train_samples = []
for example in dataset["train"]:
    query = example["query"]
    for doc in example["documents"]:
        train_samples.append((query, doc["text"], doc["relevance"]/5.0))  # Normalize to 0-1

# Train a cross-encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1)
model.fit(
    train_samples=train_samples,
    epochs=1,
    batch_size=16
)
```"""
    }
    
    # Create README content
    readme_content = f"""# Sheikh Mujibur Rahman {task_title} Dataset

{task_descriptions.get(task, "")}

## Dataset Format

This dataset follows the HuggingFace {task_title} format:
{format_explanations.get(task, "")}

## Data Sources

The data is derived from:
1. **Prothom Alo** - One of Bangladesh's leading Bangla newspapers
2. **Bangla Wikipedia** - The Bangla language version of Wikipedia

## Dataset Structure

The dataset includes:
- Training split (`train.jsonl`)
- Test/validation split (`test.jsonl` or `validation.jsonl`)
- Metadata with dataset information

## Usage Example

{usage_examples.get(task, "")}

## Citation

If you use this dataset in your research or applications, please cite it as:

```
@dataset{{mujib71_{task}_dataset,
  author       = {{Likhon Sheikh}},
  title        = {{Sheikh Mujibur Rahman {task_title} Dataset}},
  month        = may,
  year         = 2025,
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

## Related Datasets

See also our comprehensive NLP dataset with more detailed information:
- [Sheikh Mujibur Rahman Bangla NLP Dataset](https://huggingface.co/datasets/likhonsheikhdev/mujib71_datasets)
"""
    
    # Write task README to a temporary file
    temp_filename = f"{task}_readme.md"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Upload to HuggingFace
    print(f"Uploading README.md for {task} task...")
    api.upload_file(
        path_or_fileobj=temp_filename,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {task_title} dataset documentation"
    )
    
    # Remove temporary file
    os.remove(temp_filename)
    return True

if __name__ == "__main__":
    upload_dataset_to_huggingface()
