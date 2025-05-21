import os
import json
import random
import pandas as pd

def create_llm_dataset():
    """Create a dataset for LLM fine-tuning using all available data sources."""
    print("Creating LLM dataset for fine-tuning...")
    
    # Ensure the dataset directory exists
    os.makedirs("dataset/task_specific/llm/pretrain", exist_ok=True)
    os.makedirs("dataset/task_specific/llm/instruct", exist_ok=True)
    os.makedirs("dataset/task_specific/llm/chat", exist_ok=True)
    
    # Load the original data if it exists
    if not os.path.exists("dataset/mujib71_data.json"):
        print("Error: dataset/mujib71_data.json not found. Please run the scraper.py script first.")
        return False
    
    with open("dataset/mujib71_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract news articles and Wikipedia data
    news_articles = data.get("news_articles", [])
    wiki_data = data.get("wikipedia", [])
    
    # Create datasets
    create_pretrain_dataset(news_articles, wiki_data)
    create_instruction_dataset(news_articles, wiki_data)
    create_chat_dataset(news_articles, wiki_data)
    
    print("LLM dataset creation completed!")
    return True

def create_pretrain_dataset(news_articles, wiki_data):
    """Create a pretrain dataset with raw text for continued pretraining"""
    print("Creating pretrain dataset...")
    
    # Collect all text content
    all_text = []
    
    # Add news articles
    for article in news_articles:
        title = article.get("title", "")
        content = article.get("content", "")
        if title and content:
            all_text.append(f"# {title}\n\n{content}\n\n")
    
    # Add Wikipedia content
    for wiki_page in wiki_data:
        title = wiki_page.get("title", "")
        sections = wiki_page.get("sections", [])
        
        wiki_content = [f"# {title}\n\n"]
        for section in sections:
            section_title = section.get("title", "")
            section_content = section.get("content", "")
            if section_title and section_content:
                wiki_content.append(f"## {section_title}\n\n{section_content}\n\n")
        
        all_text.append("".join(wiki_content))
    
    # Write to file
    with open("dataset/task_specific/llm/pretrain/bangla_text.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    
    print(f"Created pretrain dataset with {len(all_text)} text chunks")

def create_instruction_dataset(news_articles, wiki_data):
    """Create an instruction dataset in Alpaca format for instruction fine-tuning"""
    print("Creating instruction dataset...")
    
    instruction_examples = []
    
    # Create factual Q&A from news articles
    for article in news_articles:
        title = article.get("title", "")
        content = article.get("content", "")
        
        if not title or not content:
            continue
        
        # Create an instruction example
        example = {
            "instruction": f"একটি সংক্ষিপ্ত সারাংশ লিখুন: '{title}'",
            "input": "",
            "output": content[:500] + "..." if len(content) > 500 else content
        }
        
        instruction_examples.append(example)
    
    # Add general knowledge instructions
    general_instructions = [
        {
            "instruction": "শেখ মুজিবুর রহমান কে ছিলেন?",
            "input": "",
            "output": "শেখ মুজিবুর রহমান (১৯২০-১৯৭৫) বাংলাদেশের জাতির জনক। তিনি পাকিস্তানের বিরুদ্ধে বাংলাদেশের স্বাধীনতা আন্দোলনের নেতৃত্ব দেন।"
        },
        {
            "instruction": "বাংলাদেশ কীভাবে স্বাধীনতা অর্জন করেছিল?",
            "input": "",
            "output": "বাংলাদেশ ১৯৭১ সালের ১৬ ডিসেম্বর পাকিস্তান থেকে স্বাধীনতা অর্জন করে। নয় মাসের রক্তক্ষয়ী মুক্তিযুদ্ধের পর স্বাধীনতা অর্জিত হয়।"
        }
    ]
    
    instruction_examples.extend(general_instructions)
    
    # Split into training and test sets
    random.shuffle(instruction_examples)
    train_size = int(len(instruction_examples) * 0.8)
    train_examples = instruction_examples[:train_size]
    test_examples = instruction_examples[train_size:]
    
    # Save as JSONL files
    with open("dataset/task_specific/llm/instruct/train.jsonl", "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    with open("dataset/task_specific/llm/instruct/test.jsonl", "w", encoding="utf-8") as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Created instruction dataset with {len(train_examples)} training and {len(test_examples)} test examples")

def create_chat_dataset(news_articles, wiki_data):
    """Create a chat dataset with multi-turn conversations"""
    print("Creating chat dataset...")
    
    # Create system messages
    system_message = "আপনি একটি সহায়ক এআই অ্যাসিস্ট্যান্ট যিনি বাংলাদেশের ইতিহাস এবং শেখ মুজিবুর রহমানের জীবনী সম্পর্কে বিশেষজ্ঞ।"
    
    # Create chat examples
    chat_examples = []
    
    # Process news articles into chat format
    for article in news_articles:
        title = article.get("title", "")
        content = article.get("content", "")
        
        if not title or not content:
            continue
        
        # Create a chat example
        chat_example = {
            "messages": [
                {"role": "System", "content": system_message},
                {"role": "User", "content": f"'{title}' প্রবন্ধের সারাংশ দিন।"},
                {"role": "Chatbot", "content": content[:1000] + "..." if len(content) > 1000 else content}
            ]
        }
        chat_examples.append(chat_example)
    
    # Create basic multi-turn conversations
    multi_turn_conversations = [
        {
            "messages": [
                {"role": "System", "content": system_message},
                {"role": "User", "content": "শেখ মুজিবুর রহমান কে ছিলেন?"},
                {"role": "Chatbot", "content": "শেখ মুজিবুর রহমান (১৯২০-১৯৭৫) বাংলাদেশের জাতির জনক। তিনি পাকিস্তানের বিরুদ্ধে বাংলাদেশের স্বাধীনতা আন্দোলনের নেতৃত্ব দেন।"},
                {"role": "User", "content": "মুক্তিযুদ্ধে তার ভূমিকা কি ছিল?"},
                {"role": "Chatbot", "content": "তিনি ১৯৭১ সালের ৭ মার্চে একটি গুরুত্বপূর্ণ ভাষণ দেন যা বাংলাদেশের স্বাধীনতার প্রতি মানুষকে অনুপ্রাণিত করে। যুদ্ধ চলাকালীন তিনি পাকিস্তানে বন্দি ছিলেন।"}
            ]
        }
    ]
    
    chat_examples.extend(multi_turn_conversations)
    
    # Split into training and test sets
    random.shuffle(chat_examples)
    train_size = int(len(chat_examples) * 0.8)
    train_examples = chat_examples[:train_size]
    test_examples = chat_examples[train_size:]
    
    # Save as JSONL files
    with open("dataset/task_specific/llm/chat/train.jsonl", "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    with open("dataset/task_specific/llm/chat/test.jsonl", "w", encoding="utf-8") as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Created chat dataset with {len(train_examples)} training and {len(test_examples)} test examples")

if __name__ == "__main__":
    create_llm_dataset()
