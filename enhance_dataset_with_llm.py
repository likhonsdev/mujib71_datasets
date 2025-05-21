import os
import json
import random
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

# Load API keys from environment or set them if provided
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBFhERyEMxIWk_hsWOKU-Vs0stwocn0h0g")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")

def setup_argument_parser():
    """Setup command-line arguments"""
    parser = argparse.ArgumentParser(description="Enhance datasets using LLM APIs")
    parser.add_argument("--dataset-type", choices=["chat", "instruct", "pretrain", "all"], 
                        default="all", help="Type of dataset to enhance")
    parser.add_argument("--model", choices=["groq", "gemini"], 
                        default="gemini", help="LLM model to use for enhancement")
    parser.add_argument("--examples", type=int, default=10, 
                        help="Number of new examples to generate for each dataset")
    return parser

def load_existing_dataset(dataset_type):
    """Load existing dataset from disk"""
    print(f"Loading existing {dataset_type} dataset...")
    
    datasets = {}
    
    # Path to dataset files depends on dataset type
    if dataset_type in ["chat", "all"]:
        try:
            chat_path = "dataset/task_specific/chat/train.jsonl"
            if os.path.exists(chat_path):
                datasets["chat"] = []
                with open(chat_path, "r", encoding="utf-8") as f:
                    for line in f:
                        datasets["chat"].append(json.loads(line))
                print(f"Loaded {len(datasets['chat'])} chat examples")
        except Exception as e:
            print(f"Error loading chat dataset: {e}")
    
    if dataset_type in ["instruct", "all"]:
        try:
            instruct_path = "dataset/task_specific/llm/instruct/train.jsonl"
            if os.path.exists(instruct_path):
                datasets["instruct"] = []
                with open(instruct_path, "r", encoding="utf-8") as f:
                    for line in f:
                        datasets["instruct"].append(json.loads(line))
                print(f"Loaded {len(datasets['instruct'])} instruction examples")
        except Exception as e:
            print(f"Error loading instruction dataset: {e}")
    
    # Only load dataset metadata for pretrain, as the actual content is large
    if dataset_type in ["pretrain", "all"]:
        try:
            pretrain_path = "dataset/task_specific/llm/pretrain/bangla_text.txt"
            if os.path.exists(pretrain_path):
                # Just check if the file exists, don't load full content
                file_size = os.path.getsize(pretrain_path)
                datasets["pretrain"] = {"exists": True, "size": file_size}
                print(f"Found pretrain dataset ({file_size / 1024:.2f} KB)")
        except Exception as e:
            print(f"Error checking pretrain dataset: {e}")
    
    # Load original data for content references
    try:
        original_data_path = "dataset/mujib71_data.json"
        if os.path.exists(original_data_path):
            with open(original_data_path, "r", encoding="utf-8") as f:
                datasets["original"] = json.load(f)
            print("Loaded original dataset for reference")
    except Exception as e:
        print(f"Error loading original data: {e}")
        datasets["original"] = {"news_articles": [], "wikipedia": []}
    
    return datasets

def setup_llm_client(model_name):
    """Set up the LLM client based on model choice"""
    if model_name == "groq":
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            return client, "llama3-8b-8192"
        except ImportError:
            print("Groq Python package not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "groq"])
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            return client, "llama3-8b-8192"
    else:  # gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-pro')
            return model, "gemini-1.5-pro"
        except ImportError:
            print("Google GenerativeAI package not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "google-generativeai"])
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-pro')
            return model, "gemini-1.5-pro"

def call_llm_api(client, model_name, prompt, model_type):
    """Call the appropriate LLM API based on model_type"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if model_type == "groq":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048
                )
                return response.choices[0].message.content
            else:  # gemini
                response = client.generate_content(prompt)
                return response.text
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (attempt + 1)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached, skipping this prompt")
                return None

def generate_chat_examples(client, model_name, model_type, original_data, num_examples=5):
    """Generate new chat examples using the LLM API"""
    print(f"Generating {num_examples} new chat examples...")
    
    # Extract topics and information from original data
    topics = []
    
    # Extract topics from news articles
    for article in original_data.get("news_articles", [])[:10]:  # Limit to 10 for efficiency
        title = article.get("title", "")
        if title:
            topics.append({"title": title, "type": "article"})
    
    # Extract topics from Wikipedia sections
    for wiki in original_data.get("wikipedia", []):
        for section in wiki.get("sections", [])[:10]:  # Limit to 10 sections for efficiency
            section_title = section.get("title", "")
            if section_title and section_title != "Introduction":
                topics.append({"title": section_title, "type": "wiki_section"})
    
    examples = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Prepare prompts for all examples
        futures = []
        for _ in range(num_examples):
            # Select a random topic
            if topics:
                topic = random.choice(topics)
                
                if topic["type"] == "article":
                    prompt_template = f"""
Create a multi-turn conversation between a user and an AI assistant about "{topic['title']}".
The conversation should be in Bengali language (Bangla) and include at least 3 turns (user-assistant exchanges).
The conversation should be related to Sheikh Mujibur Rahman, the founding father of Bangladesh.
Format the output as valid JSON with the following structure:
{{
  "messages": [
    {{"role": "System", "content": "[system message in Bengali]"}},
    {{"role": "User", "content": "[first user message in Bengali]"}},
    {{"role": "Chatbot", "content": "[first assistant response in Bengali]"}},
    {{"role": "User", "content": "[second user message in Bengali]"}},
    {{"role": "Chatbot", "content": "[second assistant response in Bengali]"}},
    ...
  ]
}}
Make sure the system message clearly sets context about Sheikh Mujibur Rahman.
Only return the JSON without any additional text or explanation.
"""
                else:  # wiki_section
                    prompt_template = f"""
Create a multi-turn conversation between a user and an AI assistant about "{topic['title']}" related to Sheikh Mujibur Rahman.
The conversation should be in Bengali language (Bangla) and include at least 3 turns (user-assistant exchanges).
Format the output as valid JSON with the following structure:
{{
  "messages": [
    {{"role": "System", "content": "[system message in Bengali]"}},
    {{"role": "User", "content": "[first user message in Bengali]"}},
    {{"role": "Chatbot", "content": "[first assistant response in Bengali]"}},
    {{"role": "User", "content": "[second user message in Bengali]"}},
    {{"role": "Chatbot", "content": "[second assistant response in Bengali]"}},
    ...
  ]
}}
Make sure the system message clearly sets context about Sheikh Mujibur Rahman.
Only return the JSON without any additional text or explanation.
"""
                
            # Submit to thread pool
            futures.append(executor.submit(call_llm_api, client, model_name, prompt_template, model_type))
        
        # Collect results
        for future in futures:
            result = future.result()
            if result:
                try:
                    # Clean up the response to ensure it's valid JSON
                    result = result.strip()
                    if result.startswith("```json"):
                        result = result[7:]
                    if result.endswith("```"):
                        result = result[:-3]
                    result = result.strip()
                    
                    # Parse JSON
                    example = json.loads(result)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse generated chat example: {e}")
                    print(f"Raw output: {result[:100]}...")
    
    print(f"Successfully generated {len(examples)} new chat examples")
    return examples

def generate_instruction_examples(client, model_name, model_type, original_data, num_examples=5):
    """Generate new instruction examples using the LLM API"""
    print(f"Generating {num_examples} new instruction examples...")
    
    # Extract topics from original data
    topics = []
    
    # Extract topics from news articles
    for article in original_data.get("news_articles", [])[:10]:
        title = article.get("title", "")
        if title:
            topics.append({"title": title, "type": "article"})
    
    # Extract topics from Wikipedia sections
    for wiki in original_data.get("wikipedia", []):
        for section in wiki.get("sections", [])[:10]:
            section_title = section.get("title", "")
            if section_title and section_title != "Introduction":
                topics.append({"title": section_title, "type": "wiki_section"})
    
    examples = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Prepare prompts for all examples
        futures = []
        for _ in range(num_examples):
            # Select a random topic
            if topics:
                topic = random.choice(topics)
                
                prompt_template = f"""
Create an instruction example about "{topic['title']}" related to Sheikh Mujibur Rahman in Bangla (Bengali) language.
The example should be formatted as a JSON object with the following structure:
{{
  "instruction": "[A clear instruction in Bangla language asking about {topic['title']}]",
  "input": "", 
  "output": "[A comprehensive response in Bangla language about {topic['title']} related to Sheikh Mujibur Rahman]"
}}
Make sure the instruction is clear and specific. The output should be comprehensive and informative.
The "input" field should be an empty string.
Only return the JSON without any additional text or explanation.
"""
                
                # Submit to thread pool
                futures.append(executor.submit(call_llm_api, client, model_name, prompt_template, model_type))
        
        # Collect results
        for future in futures:
            result = future.result()
            if result:
                try:
                    # Clean up the response to ensure it's valid JSON
                    result = result.strip()
                    if result.startswith("```json"):
                        result = result[7:]
                    if result.endswith("```"):
                        result = result[:-3]
                    result = result.strip()
                    
                    # Parse JSON
                    example = json.loads(result)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse generated instruction example: {e}")
                    print(f"Raw output: {result[:100]}...")
    
    print(f"Successfully generated {len(examples)} new instruction examples")
    return examples

def generate_pretrain_content(client, model_name, model_type, original_data, num_paragraphs=5):
    """Generate additional pretrain content using the LLM API"""
    print(f"Generating {num_paragraphs} new text paragraphs for pretrain dataset...")
    
    topics = [
        "Early life of Sheikh Mujibur Rahman",
        "Sheikh Mujibur Rahman's education",
        "Political career of Sheikh Mujibur Rahman",
        "Sheikh Mujibur Rahman's role in the Language Movement",
        "The Six-Point Movement",
        "Sheikh Mujibur Rahman's imprisonment",
        "7th March Speech of Sheikh Mujibur Rahman",
        "Sheikh Mujibur Rahman during the Liberation War",
        "Sheikh Mujibur Rahman's return to Bangladesh after independence",
        "Sheikh Mujibur Rahman's economic policies",
        "Foreign policy of Sheikh Mujibur Rahman",
        "Sheikh Mujibur Rahman's assassination",
        "Legacy of Sheikh Mujibur Rahman",
        "Sheikh Mujibur Rahman's family",
        "Historical significance of Sheikh Mujibur Rahman",
        "Cultural impact of Sheikh Mujibur Rahman",
        "Sheikh Mujibur Rahman's vision for Bangladesh",
        "Bangabandhu: The Friend of Bengal",
        "Sheikh Mujibur Rahman's speeches and writings",
        "International recognition of Sheikh Mujibur Rahman"
    ]
    
    paragraphs = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Prepare prompts for all paragraphs
        futures = []
        for _ in range(num_paragraphs):
            topic = random.choice(topics)
            
            prompt_template = f"""
Write a detailed paragraph about "{topic}" in Bengali (Bangla) language.
The content should be factually accurate, informative, and suitable for inclusion in a language model's pretraining dataset.
Include historical context, specific dates, names, and events when relevant.
Write 3-5 paragraphs, with each paragraph focusing on a different aspect of the topic.
Use formal and scholarly language suitable for educational content.
"""
            
            # Submit to thread pool
            futures.append(executor.submit(call_llm_api, client, model_name, prompt_template, model_type))
        
        # Collect results
        for future in futures:
            result = future.result()
            if result:
                # Remove any markdown formatting or extra text
                if result.startswith("```"):
                    result = result.split("```")[1]
                    if result.startswith("markdown") or result.startswith("text"):
                        result = result.split("\n", 1)[1]
                
                paragraphs.append(result)
    
    # Combine all paragraphs with proper formatting
    combined_text = "\n\n# Generated Content\n\n" + "\n\n".join(paragraphs)
    print(f"Successfully generated {len(paragraphs)} new paragraphs for pretrain dataset")
    return combined_text

def save_enhanced_datasets(datasets, new_content):
    """Save the enhanced datasets back to disk"""
    print("Saving enhanced datasets...")
    
    # Create necessary directories
    os.makedirs("dataset/task_specific/llm/instruct", exist_ok=True)
    os.makedirs("dataset/task_specific/llm/chat", exist_ok=True)
    os.makedirs("dataset/task_specific/llm/pretrain", exist_ok=True)
    os.makedirs("dataset/task_specific/chat", exist_ok=True)
    
    # Save chat dataset
    if "chat" in new_content and new_content["chat"]:
        chat_path = "dataset/task_specific/chat/train.jsonl"
        if os.path.exists(chat_path):
            # Load existing examples first
            existing_examples = []
            with open(chat_path, "r", encoding="utf-8") as f:
                for line in f:
                    existing_examples.append(json.loads(line))
            
            # Add new examples
            existing_examples.extend(new_content["chat"])
            
            # Save back to file
            with open(chat_path, "w", encoding="utf-8") as f:
                for example in existing_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            print(f"Updated chat dataset with {len(new_content['chat'])} new examples")
        else:
            # Create new file
            with open(chat_path, "w", encoding="utf-8") as f:
                for example in new_content["chat"]:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            print(f"Created new chat dataset with {len(new_content['chat'])} examples")
    
    # Save instruction dataset
    if "instruct" in new_content and new_content["instruct"]:
        instruct_path = "dataset/task_specific/llm/instruct/train.jsonl"
        if os.path.exists(instruct_path):
            # Load existing examples first
            existing_examples = []
            with open(instruct_path, "r", encoding="utf-8") as f:
                for line in f:
                    existing_examples.append(json.loads(line))
            
            # Add new examples
            existing_examples.extend(new_content["instruct"])
            
            # Save back to file
            with open(instruct_path, "w", encoding="utf-8") as f:
                for example in existing_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            print(f"Updated instruction dataset with {len(new_content['instruct'])} new examples")
        else:
            # Create new file
            with open(instruct_path, "w", encoding="utf-8") as f:
                for example in new_content["instruct"]:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            print(f"Created new instruction dataset with {len(new_content['instruct'])} examples")
    
    # Save pretrain content
    if "pretrain" in new_content and new_content["pretrain"]:
        pretrain_path = "dataset/task_specific/llm/pretrain/bangla_text.txt"
        if os.path.exists(pretrain_path):
            # Append to existing file
            with open(pretrain_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + new_content["pretrain"])
            
            print("Appended new content to pretrain dataset")
        else:
            # Create new file
            with open(pretrain_path, "w", encoding="utf-8") as f:
                f.write(new_content["pretrain"])
            
            print("Created new pretrain dataset")
    
    print("Successfully saved all enhanced datasets")

def main():
    """Main function to run the dataset enhancement process"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    print(f"=== Enhancing Sheikh Mujibur Rahman Dataset using {args.model.upper()} ===")
    
    # Load existing datasets
    datasets = load_existing_dataset(args.dataset_type)
    
    # Setup LLM client
    client, model_name = setup_llm_client(args.model)
    print(f"Using {model_name} model for enhancement")
    
    # Generate new content based on dataset type
    new_content = {}
    
    if args.dataset_type in ["chat", "all"] and "original" in datasets:
        new_content["chat"] = generate_chat_examples(
            client, model_name, args.model, datasets["original"], num_examples=args.examples
        )
    
    if args.dataset_type in ["instruct", "all"] and "original" in datasets:
        new_content["instruct"] = generate_instruction_examples(
            client, model_name, args.model, datasets["original"], num_examples=args.examples
        )
    
    if args.dataset_type in ["pretrain", "all"] and "original" in datasets:
        new_content["pretrain"] = generate_pretrain_content(
            client, model_name, args.model, datasets["original"], num_paragraphs=args.examples
        )
    
    # Save the enhanced datasets
    save_enhanced_datasets(datasets, new_content)
    
    print("\n=== Dataset Enhancement Complete! ===")
    print(f"Enhanced datasets with {args.examples} new examples each using {args.model} model.")
    print("The datasets are now ready for use in training or fine-tuning LLMs.")

if __name__ == "__main__":
    main()
