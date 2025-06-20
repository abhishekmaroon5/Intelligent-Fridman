#!/usr/bin/env python3
"""
Comprehensive Dataset Creator for Lex Fridman Chatbot
Processes all available transcripts (JSON and TXT formats) into a unified training dataset
"""

import json
import os
import re
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCreator:
    def __init__(self):
        self.data_dir = Path("data")
        self.transcripts_dir = self.data_dir / "transcripts"
        self.txt_transcripts_dir = Path("trascripts")  # Note: typo in original folder name
        self.processed_dir = Path("processed_data")
        self.processed_dir.mkdir(exist_ok=True)
        
    def process_json_transcript(self, file_path: Path) -> List[Dict]:
        """Process JSON transcript file from Tim Sweeney episode"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        title = data.get('title', 'Unknown Episode')
        transcript_text = data.get('transcript', '')
        
        logger.info(f"Processing JSON: {title}")
        
        # Split transcript into conversational chunks
        conversations = self._split_into_conversations(transcript_text, title)
        return conversations
    
    def process_txt_transcript(self, file_path: Path) -> List[Dict]:
        """Process TXT transcript files (NoteGPT format)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from filename
        filename = file_path.stem
        if "NoteGPT_" in filename:
            title = filename.replace("NoteGPT_", "").replace("_", " ")
        else:
            title = filename
        
        logger.info(f"Processing TXT: {title}")
        
        # Parse the timestamp-based format
        conversations = self._parse_timestamped_transcript(content, title)
        return conversations
    
    def _parse_timestamped_transcript(self, content: str, title: str) -> List[Dict]:
        """Parse timestamped transcript format from TXT files"""
        conversations = []
        
        # Split by timestamp pattern (HH:MM:SS)
        timestamp_pattern = r'\n(\d{2}:\d{2}:\d{2})\n'
        sections = re.split(timestamp_pattern, content)
        
        current_text = ""
        for i, section in enumerate(sections):
            if re.match(r'\d{2}:\d{2}:\d{2}', section):
                # This is a timestamp
                continue
            elif section.strip():
                current_text += section.strip() + " "
                
                # Create conversation chunks of reasonable size
                if len(current_text) > 1000:  # Chunk size
                    conversations.extend(self._create_conversation_pairs(current_text, title))
                    current_text = ""
        
        # Process remaining text
        if current_text.strip():
            conversations.extend(self._create_conversation_pairs(current_text, title))
        
        return conversations
    
    def _split_into_conversations(self, text: str, title: str) -> List[Dict]:
        """Split transcript text into conversational pairs"""
        # Clean the text
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        conversations = []
        
        # Create conversation pairs
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                human_text = sentences[i].strip()
                assistant_text = sentences[i + 1].strip()
                
                if len(human_text) > 10 and len(assistant_text) > 10:
                    conversations.append({
                        "instruction": f"You are a helpful AI assistant in the style of Lex Fridman. Respond thoughtfully to: {human_text}",
                        "input": "",
                        "output": assistant_text,
                        "source": title,
                        "type": "conversation"
                    })
        
        return conversations
    
    def _create_conversation_pairs(self, text: str, title: str) -> List[Dict]:
        """Create conversation pairs from text chunks"""
        conversations = []
        
        # Split by speaker indicators or natural breaks
        segments = re.split(r'[-–—]\s*', text)
        segments = [s.strip() for s in segments if s.strip() and len(s) > 20]
        
        # Create various types of training examples
        for i, segment in enumerate(segments):
            # Direct conversation style
            if i + 1 < len(segments):
                conversations.append({
                    "instruction": "You are Lex Fridman, a podcaster and AI researcher. Continue this conversation naturally.",
                    "input": segment,
                    "output": segments[i + 1],
                    "source": title,
                    "type": "continuation"
                })
            
            # Question-answer style
            if "?" in segment:
                parts = segment.split("?", 1)
                if len(parts) == 2 and len(parts[1].strip()) > 10:
                    conversations.append({
                        "instruction": "Answer this question in the style of Lex Fridman's podcast conversations.",
                        "input": parts[0].strip() + "?",
                        "output": parts[1].strip(),
                        "source": title,
                        "type": "qa"
                    })
            
            # Topic discussion
            if len(segment) > 100:
                conversations.append({
                    "instruction": "Discuss this topic in depth, as Lex Fridman would in his podcast.",
                    "input": f"What are your thoughts on: {segment[:100]}...",
                    "output": segment,
                    "source": title,
                    "type": "discussion"
                })
        
        return conversations
    
    def create_unified_dataset(self) -> Dict:
        """Create unified dataset from all transcript sources"""
        all_conversations = []
        
        # Process JSON transcript (Tim Sweeney)
        json_files = list(self.transcripts_dir.glob("*.json"))
        for json_file in json_files:
            try:
                conversations = self.process_json_transcript(json_file)
                all_conversations.extend(conversations)
                logger.info(f"Added {len(conversations)} conversations from {json_file.name}")
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        # Process TXT transcripts
        txt_files = list(self.txt_transcripts_dir.glob("*.txt"))
        for txt_file in txt_files:
            try:
                conversations = self.process_txt_transcript(txt_file)
                all_conversations.extend(conversations)
                logger.info(f"Added {len(conversations)} conversations from {txt_file.name}")
            except Exception as e:
                logger.error(f"Error processing {txt_file}: {e}")
        
        # Create dataset structure
        dataset = {
            "metadata": {
                "total_conversations": len(all_conversations),
                "sources": [
                    "Tim Sweeney: Fortnite, Unreal Engine, and the Future of Gaming",
                    "Sara Walker: Physics of Life, Time, Complexity, and Aliens", 
                    "Mark Zuckerberg: Future of AI at Meta, Facebook, Instagram, and WhatsApp",
                    "Israel-Palestine Debate: Finkelstein, Destiny, M. Rabbani & Benny Morris",
                    "DeepSeek, China, OpenAI, NVIDIA, xAI, TSMC, Stargate, and AI Megaclusters",
                    "Dario Amodei: Anthropic CEO on Claude, AGI & the Future of AI & Humanity"
                ],
                "topics": [
                    "Technology & Gaming", "Programming & Software Engineering", 
                    "AI & Machine Learning", "Physics & Science", "Philosophy",
                    "Business & Entrepreneurship", "Social Media & Tech Companies",
                    "Geopolitics & International Relations", "Future of Humanity"
                ],
                "total_episodes": 6,
                "format_version": "1.0"
            },
            "conversations": all_conversations
        }
        
        return dataset
    
    def analyze_dataset(self, dataset: Dict) -> Dict:
        """Analyze the created dataset for statistics"""
        conversations = dataset["conversations"]
        
        # Calculate statistics
        total_chars = sum(len(conv["input"]) + len(conv["output"]) for conv in conversations)
        avg_input_length = sum(len(conv["input"]) for conv in conversations) / len(conversations)
        avg_output_length = sum(len(conv["output"]) for conv in conversations) / len(conversations)
        
        # Count by source
        source_counts = {}
        for conv in conversations:
            source = conv.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Count by type
        type_counts = {}
        for conv in conversations:
            conv_type = conv.get("type", "Unknown")
            type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
        
        analysis = {
            "total_conversations": len(conversations),
            "total_characters": total_chars,
            "estimated_words": total_chars // 5,  # Rough estimate
            "average_input_length": round(avg_input_length, 2),
            "average_output_length": round(avg_output_length, 2),
            "source_distribution": source_counts,
            "type_distribution": type_counts
        }
        
        return analysis
    
    def save_dataset(self, dataset: Dict, analysis: Dict):
        """Save the dataset and analysis to files"""
        # Save main dataset
        dataset_file = self.processed_dir / "unified_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Save analysis
        analysis_file = self.processed_dir / "dataset_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Save training format (simplified for transformers)
        training_data = []
        for conv in dataset["conversations"]:
            training_data.append({
                "text": f"### Human: {conv['input']}\n### Assistant: {conv['output']}"
            })
        
        training_file = self.processed_dir / "training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {dataset_file}")
        logger.info(f"Analysis saved to {analysis_file}")
        logger.info(f"Training data saved to {training_file}")

def main():
    """Main execution function"""
    print("🚀 Creating Unified Lex Fridman Chatbot Dataset")
    print("=" * 50)
    
    creator = DatasetCreator()
    
    # Create dataset
    print("📊 Processing all transcript files...")
    dataset = creator.create_unified_dataset()
    
    # Analyze dataset
    print("🔍 Analyzing dataset...")
    analysis = creator.analyze_dataset(dataset)
    
    # Print analysis
    print("\n📈 Dataset Analysis:")
    print(f"Total Conversations: {analysis['total_conversations']:,}")
    print(f"Total Characters: {analysis['total_characters']:,}")
    print(f"Estimated Words: {analysis['estimated_words']:,}")
    print(f"Avg Input Length: {analysis['average_input_length']} chars")
    print(f"Avg Output Length: {analysis['average_output_length']} chars")
    
    print("\n📚 Source Distribution:")
    for source, count in analysis['source_distribution'].items():
        print(f"  {source}: {count} conversations")
    
    print("\n🎯 Type Distribution:")
    for conv_type, count in analysis['type_distribution'].items():
        print(f"  {conv_type}: {count} conversations")
    
    # Save everything
    print("\n💾 Saving dataset files...")
    creator.save_dataset(dataset, analysis)
    
    print("\n✅ Dataset creation complete!")
    print(f"📁 Files saved in: {creator.processed_dir}")
    print("\nReady for model training! 🎉")

if __name__ == "__main__":
    main() 