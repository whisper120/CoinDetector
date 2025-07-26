# Rare Coin Detection using AI

## Overview

The field of numismatics, or coin collecting, has long relied on expert analysis to identify rare coins, with rarity often determined by factors such as mint year, production errors, or limited circulation. However, this manual identification process can be time-consuming, prone to error, and inaccessible to those without specialized knowledge.

Recent advancements in artificial intelligence (AI) and computer vision offer a powerful solution to these challenges by automating the recognition and classification of rare coins. Existing solutions work with images of single coins expected to be rare. Nevertheless, many non-professional coin collections are composed of coins collected from circulation or removed from circulation within the last 100 years. Identifying coins one-by-one in such collections is tedious and inefficient with current tools.

This project focuses on developing an AI-based system for detecting **rare Israeli coins** in piles (placers) of coins, with a specific emphasis on identifying distinctions based on **minting dates**. Using the **YOLO (You Only Look Once)** object detection framework and the **RF-DETR (Roboflow-developed model based on the Detection Transformer)**, the system can accurately detect and classify rare coins from common ones via static images.

---

## Getting Started

### 🔧 Installation

1. Clone the repository
2. Install the required dependencies:
	pip install -r requirements.txt
3. Because the RF DETR model is over 100 mb , you need to download it from the releases section, just download it and put it into models folder.
## 🖼️ Using the GUI

To use the graphical user interface (GUI), simply run:

python GUI_hybrid.py


This will launch a user-friendly interface for loading images or accessing live camera input for the coin detection.

## 📄 Additional Documentation

For more detailed instructions on system usage, model structure, and dataset handling, please refer to the `Coin Detection phase B.pdf` file included in the repository.
