import os
import xml.etree.ElementTree as ET
import json
import glob
import random

def parse_xml_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract URL from the Document tag attribute
        source_url = root.get('url') or ""
        
        # Extract common metadata
        focus = root.findtext('Focus')
        
        qa_pairs = []
        # QAPairs
        for qa_pair in root.findall('QAPairs/QAPair'):
            question_elem = qa_pair.find('Question')
            if question_elem is not None:
                question = question_elem.text
                answer = qa_pair.findtext('Answer')
                
                # Valid if it has an answer
                if question and answer and answer.strip():
                    qa_item = {
                        "Question": question.strip(),
                        "Answer": answer.strip(),
                        "Source": source_url.strip()
                    }
                    qa_pairs.append(qa_item)
                
        return qa_pairs
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def main():
    raw_data_dir = "./fine-tune/raw-data"
    mlx_data_dir = "./fine-tune/mlx-data"
    
    # Ensure mlx-data directory exists
    os.makedirs(mlx_data_dir, exist_ok=True)
    
    # Use glob to find all xml files recursively
    xml_files = glob.glob(os.path.join(raw_data_dir, "**/*.xml"), recursive=True)
    print(f"Found {len(xml_files)} XML files.")
    
    all_data = []
    for i, xml_file in enumerate(xml_files):
        if i % 500 == 0:
            print(f"Processing file {i} of {len(xml_files)}...")
        qa_items = parse_xml_file(xml_file)
        all_data.extend(qa_items)
        
    print(f"Extracted {len(all_data)} valid question-answer pairs.")
    
    # Prepare MLX format (.jsonl)
    # Construction: "Question: ...\nAnswer: ..."
    # We'll do a simple 90/10 split for train/valid
    random.seed(42)
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    valid_data = all_data[split_idx:]
    
    def save_jsonl(data_list, filename):
        with open(os.path.join(mlx_data_dir, filename), 'w', encoding='utf-8') as f:
            for item in data_list:
                # We save the full metadata so a retriever can use it later
                # and include it in the text field for potential fine-tuning
                text = f"Question: {item['Question']}\nAnswer: {item['Answer']}\nSource: {item['Source']}"
                f.write(json.dumps({"text": text, "source": item['Source']}) + "\n")

    save_jsonl(train_data, "train.jsonl")
    save_jsonl(valid_data, "valid.jsonl")
    
    print(f"Successfully saved MLX formatted data (train.jsonl, valid.jsonl) to {mlx_data_dir}")

if __name__ == "__main__":
    main()
