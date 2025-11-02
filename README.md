# Introduction
This repository contains the complete implementation for the study "Cognitive Diversity in Artificial Minds - Investigating the Performance of Large Language Models Under Memory Constraints". this research systematically investigates how transformer-based language models dynamically adjust their attention patterns when processing sequences that exceed their effective memory capacity. Through controlled experiments across diverse model architectures—including causal (GPT-2, Phi-2, GPT-Neo) and masked (DistilBERT) language models—we demonstrate that models exhibit distinct attention entropy signatures when operating beyond their memory limits, revealing fundamental differences in how various architectures manage computational constraints during sequence processing.

# How to Run the Code
- Download/Clone the directory above
- Install dependencies whith `pip install -r requirements.txt`
- Run the file named `main.py`
