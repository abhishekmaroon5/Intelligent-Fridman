#!/usr/bin/env python3
"""
Sample Dataset Viewer
Shows sample conversations from the unified 6-transcript dataset
"""

import json
import random
from pathlib import Path

def view_samples():
    """Display sample conversations from different sources and types"""
    
    print("🔍 Sample Conversations from 6-Transcript Dataset")
    print("=" * 60)
    
    # Load dataset
    with open("processed_data/unified_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = data["conversations"]
    
    # Group by source
    by_source = {}
    for conv in conversations:
        source = conv["source"]
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(conv)
    
    # Show samples from each source
    for source, convs in by_source.items():
        print(f"\n📚 **{source}**")
        print("-" * 50)
        
        # Show 2 random samples
        samples = random.sample(convs, min(2, len(convs)))
        
        for i, conv in enumerate(samples, 1):
            print(f"\n🔹 Sample {i} ({conv['type']}):")
            
            if conv['input'].strip():
                print(f"🗣️  Input: {conv['input'][:200]}{'...' if len(conv['input']) > 200 else ''}")
            else:
                print(f"📋 Instruction: {conv['instruction'][:200]}{'...' if len(conv['instruction']) > 200 else ''}")
            
            print(f"🤖 Output: {conv['output'][:300]}{'...' if len(conv['output']) > 300 else ''}")
    
    # Show type distribution
    print(f"\n\n📊 **Dataset Statistics:**")
    print(f"Total Conversations: {len(conversations):,}")
    
    type_counts = {}
    for conv in conversations:
        conv_type = conv.get('type', 'Unknown')
        type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
    
    print("\n🎯 **Type Distribution:**")
    for conv_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(conversations)) * 100
        print(f"  {conv_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📚 **Source Distribution:**")
    for source, convs in sorted(by_source.items(), key=lambda x: len(x[1]), reverse=True):
        percentage = (len(convs) / len(conversations)) * 100
        print(f"  {len(convs):,} conversations ({percentage:.1f}%) - {source.split('|')[0].strip()}")

if __name__ == "__main__":
    view_samples() 