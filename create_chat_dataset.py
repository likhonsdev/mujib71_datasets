import json
import os
import random

def create_system_message():
    """Create a system message for the chat dataset"""
    system_messages = [
        "You are a helpful AI assistant with expertise in Bangladeshi history and the life of Sheikh Mujibur Rahman.",
        "You are a knowledgeable virtual assistant specializing in the history of Bangladesh and its founding father Sheikh Mujibur Rahman.",
        "You are a virtual guide with extensive knowledge about Sheikh Mujibur Rahman and the independence movement of Bangladesh.",
        "You are a digital curator of Bangladeshi history, particularly focused on the contributions of Sheikh Mujibur Rahman.",
        "You are an AI with comprehensive knowledge about Bangladesh's fight for independence and Sheikh Mujibur Rahman's leadership."
    ]
    return random.choice(system_messages)

def create_chat_dataset():
    """
    Create a Hugging Face-compatible chat dataset from the existing Mujib71 data
    """
    print("Creating HuggingFace-compatible chat dataset...")
    
    if not os.path.exists("dataset/mujib71_data.json"):
        print("Error: dataset/mujib71_data.json not found. Please run the scraper.py script first.")
        return False
    
    # Load the existing data
    with open("dataset/mujib71_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract news articles and Wikipedia data
    news_articles = data.get("news_articles", [])
    wiki_data = data.get("wikipedia", [])
    
    chat_examples = []
    
    # Process news articles into chat format
    for article in news_articles:
        title = article.get("title", "")
        content = article.get("content", "")
        
        if not title or not content:
            continue
        
        # Create questions about this news article
        questions = [
            f"Tell me about the news article titled '{title}'.",
            f"What information is provided in the article '{title}'?",
            f"Can you summarize the article about '{title}'?",
            f"What are the key points discussed in the news article '{title}'?",
            f"I'm interested in information about '{title}'. What does the article say?"
        ]
        
        # Create a chat example with random question
        chat_example = {
            "messages": [
                {"role": "System", "content": create_system_message()},
                {"role": "User", "content": random.choice(questions)},
                {"role": "Chatbot", "content": content}
            ]
        }
        chat_examples.append(chat_example)
    
    # Process Wikipedia data into chat format
    for wiki_page in wiki_data:
        title = wiki_page.get("title", "")
        sections = wiki_page.get("sections", [])
        
        for section in sections:
            section_title = section.get("title", "")
            section_content = section.get("content", "")
            
            if not section_title or not section_content:
                continue
            
            # Create questions about this section
            questions = [
                f"What can you tell me about {section_title} of Sheikh Mujibur Rahman?",
                f"Provide information about {section_title} related to Sheikh Mujibur Rahman.",
                f"I'm researching about {section_title} of Sheikh Mujibur Rahman. What information can you provide?",
                f"Tell me about the {section_title} phase of Sheikh Mujibur Rahman's life.",
                f"What are the important details about {section_title} concerning Sheikh Mujibur Rahman?"
            ]
            
            # Create a chat example
            chat_example = {
                "messages": [
                    {"role": "System", "content": create_system_message()},
                    {"role": "User", "content": random.choice(questions)},
                    {"role": "Chatbot", "content": section_content}
                ]
            }
            chat_examples.append(chat_example)
    
    # Create examples without system message
    for i in range(min(len(chat_examples) // 3, 10)):  # Create some examples without system message
        if i < len(chat_examples):
            chat_examples[i]["messages"] = chat_examples[i]["messages"][1:]  # Remove system message
    
    # Create Factual Q&A examples
    factual_qa = [
        {
            "question": "When was Sheikh Mujibur Rahman born?",
            "answer": "Sheikh Mujibur Rahman was born on March 17, 1920, in Tungipara, Bengal Presidency, British India (now in Bangladesh)."
        },
        {
            "question": "What is Sheikh Mujibur Rahman known for?",
            "answer": "Sheikh Mujibur Rahman is known as the founding father of Bangladesh. He was the first President and later Prime Minister of Bangladesh. He led the independence movement against Pakistan, and is referred to as 'Bangabandhu' (Friend of Bengal)."
        },
        {
            "question": "When did Bangladesh gain independence?",
            "answer": "Bangladesh gained independence from Pakistan on December 16, 1971, after a nine-month-long Liberation War."
        },
        {
            "question": "What was the significance of the Six-Point Movement?",
            "answer": "The Six-Point Movement was a significant political movement in East Pakistan (now Bangladesh) spearheaded by Sheikh Mujibur Rahman. It called for greater autonomy and eventually became the stepping stone toward Bangladesh's independence from Pakistan."
        },
        {
            "question": "When was Sheikh Mujibur Rahman assassinated?",
            "answer": "Sheikh Mujibur Rahman was assassinated on August 15, 1975, along with most of his family members, during a military coup."
        },
        {
            "question": "What is the 7th March Speech?",
            "answer": "The 7th March Speech was a historical speech given by Sheikh Mujibur Rahman on March 7, 1971, at the Ramna Race Course (now Suhrawardy Udyan) in Dhaka. In this pivotal speech, he called for the independence of Bangladesh and is remembered for saying 'Ebarer Sangram, Muktir Sangram, Ebarer Sangram, Swadhinatar Sangram' (This struggle is for our freedom, this struggle is for our independence)."
        },
        {
            "question": "Who were Sheikh Mujibur Rahman's family members?",
            "answer": "Sheikh Mujibur Rahman's family included his wife Begum Fazilatunnessa, and their children Sheikh Hasina, Sheikh Kamal, Sheikh Jamal, Sheikh Rehana, and Sheikh Russel. Tragically, all family members except Sheikh Hasina and Sheikh Rehana (who were abroad at the time) were assassinated along with him in 1975."
        },
        {
            "question": "What was Sheikh Mujibur Rahman's role in the Language Movement?",
            "answer": "Sheikh Mujibur Rahman played a significant role in the Bengali Language Movement of 1952, which advocated for the recognition of Bengali as an official language of Pakistan. His participation and arrest during this movement strengthened his position as a leader advocating for the rights of Bengali-speaking people."
        },
        {
            "question": "What is Bangabandhu?",
            "answer": "Bangabandhu (meaning 'Friend of Bengal' in Bengali) is the honorary title given to Sheikh Mujibur Rahman. The title was conferred to him by the people of Bangladesh in recognition of his leadership in the independence movement and his contributions to the creation of Bangladesh."
        },
        {
            "question": "What is Sheikh Mujibur Rahman's political philosophy?",
            "answer": "Sheikh Mujibur Rahman's political philosophy centered around four key principles: nationalism, democracy, secularism, and socialism. These principles, together known as 'Mujibism,' formed the foundation of Bangladesh's first constitution and continue to influence the country's politics today."
        }
    ]
    
    for qa in factual_qa:
        chat_example = {
            "messages": [
                {"role": "User", "content": qa["question"]},
                {"role": "Chatbot", "content": qa["answer"]}
            ]
        }
        chat_examples.append(chat_example)
    
    # Create multi-turn conversations
    multi_turn_conversations = [
        {
            "messages": [
                {"role": "System", "content": create_system_message()},
                {"role": "User", "content": "Who was Sheikh Mujibur Rahman?"},
                {"role": "Chatbot", "content": "Sheikh Mujibur Rahman (1920-1975) was the founding father of Bangladesh. He led the country's independence movement against Pakistan and became Bangladesh's first President and later Prime Minister. He is reverently referred to as 'Bangabandhu' or Friend of Bengal."},
                {"role": "User", "content": "Tell me about his role in the liberation war."},
                {"role": "Chatbot", "content": "Sheikh Mujibur Rahman played a pivotal role in Bangladesh's Liberation War of 1971. His famous speech on March 7, 1971, effectively declared independence from Pakistan. He was arrested and imprisoned in Pakistan during the war. While imprisoned, he was symbolically declared the first President of Bangladesh. The nine-month war, fought under his inspiration and leadership (though in absentia), resulted in Bangladesh's independence on December 16, 1971."}
            ]
        },
        {
            "messages": [
                {"role": "System", "content": create_system_message()},
                {"role": "User", "content": "What was the Six-Point Movement?"},
                {"role": "Chatbot", "content": "The Six-Point Movement was a significant political movement in East Pakistan (now Bangladesh) launched by Sheikh Mujibur Rahman in 1966. The movement's demands included greater autonomy for East Pakistan, with separate currency, taxation powers, and control over foreign exchange earnings. It became a pivotal moment in the Bengali nationalist movement and laid the groundwork for eventual independence."},
                {"role": "User", "content": "How did the Pakistani government respond to this movement?"},
                {"role": "Chatbot", "content": "The Pakistani government responded harshly to the Six-Point Movement. They viewed it as a separatist agenda threatening Pakistan's unity. Sheikh Mujibur Rahman was arrested in the Agartala Conspiracy Case, accused of conspiring with India to separate East Pakistan. The government also used military force to suppress demonstrations supporting the movement. This repression further fueled Bengali nationalism and widened the divide between East and West Pakistan."}
            ]
        },
        {
            "messages": [
                {"role": "System", "content": create_system_message()},
                {"role": "User", "content": "What happened to Sheikh Mujibur Rahman after Bangladesh's independence?"},
                {"role": "Chatbot", "content": "After Bangladesh's independence, Sheikh Mujibur Rahman was released from Pakistani prison in January 1972 and returned to a hero's welcome in Bangladesh. He became the country's first Prime Minister and focused on rebuilding the war-ravaged nation. His government created Bangladesh's first constitution in 1972, based on the principles of nationalism, democracy, secularism, and socialism. Unfortunately, his leadership ended tragically when he was assassinated along with most of his family on August 15, 1975, in a military coup."},
                {"role": "User", "content": "What challenges did he face while governing Bangladesh?"},
                {"role": "Chatbot", "content": "Sheikh Mujibur Rahman faced enormous challenges while governing the newly independent Bangladesh. The country's infrastructure was devastated by the war, and millions of refugees needed repatriation. The economy was in shambles with severe food shortages. He faced political opposition from both left and right wings, and there were numerous coup attempts. International recognition and aid were urgent priorities. Despite his popularity, rebuilding a nation from scratch proved extremely difficult, especially with limited resources and the global oil crisis of the early 1970s impacting the economy further."}
            ]
        }
    ]
    
    # Add multi-turn conversations to examples
    chat_examples.extend(multi_turn_conversations)
    
    # Ensure we have enough examples (at least 16 for validation)
    if len(chat_examples) < 32:
        print(f"Warning: Only generated {len(chat_examples)} examples. HuggingFace recommends at least 32 (16 for training, 16 for validation).")
    
    # Shuffle the examples
    random.shuffle(chat_examples)
    
    # Split into training and validation sets
    split_idx = max(16, len(chat_examples) // 2)
    train_examples = chat_examples[split_idx:]
    val_examples = chat_examples[:split_idx]
    
    # Create dataset directories
    os.makedirs("dataset/task_specific/chat", exist_ok=True)
    
    # Write training examples
    with open("dataset/task_specific/chat/train.jsonl", "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Write validation examples
    with open("dataset/task_specific/chat/validation.jsonl", "w", encoding="utf-8") as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # For backward compatibility
    os.makedirs("dataset/huggingface", exist_ok=True)
    with open("dataset/huggingface/train.jsonl", "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    with open("dataset/huggingface/validation.jsonl", "w", encoding="utf-8") as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Successfully created HuggingFace chat dataset:")
    print(f"- Training examples: {len(train_examples)}")
    print(f"- Validation examples: {len(val_examples)}")
    print(f"- Files saved to: dataset/task_specific/chat/train.jsonl and dataset/task_specific/chat/validation.jsonl")
    
    return True

if __name__ == "__main__":
    create_chat_dataset()
