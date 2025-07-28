# Project: Universal Document Intelligence Engine

This repository contains the complete solution for Round 1B of the "Connecting the Dots" Hackathon. It features a robust, AI-powered pipeline designed for domain-agnostic document understanding.

## Approach Overview

Our Round 1B solution is a compact, AI-first pipeline engineered for intelligent document parsing and analysis across diverse content types—financial reports, travel guides, recipes, and beyond. It avoids brittle rule-based logic and instead relies on deep semantic reasoning and dynamic filtering to deliver precise answers under all constraints.

### 🔍 1. Universal Document Parser

We employ a layout-aware PDF parser using PyMuPDF (`fitz`) to extract clean document sections. It detects headings and body text based on font size and positioning—allowing accurate segmentation without any hardcoded rules or domain knowledge.

### 🧠 2. AI-Powered Semantic Ranking

We use the `all-mpnet-base-v2` transformer model to embed both the persona+task and extracted document sections into high-dimensional space. Cosine similarity scores between query and sections allow us to semantically rank the content. This yields relevant outputs even for nuanced or loosely-worded queries.

### 🛡️ 3. Dynamic Guardrail Filtering

To enforce hard constraints (e.g., "vegetarian only"), a final filter scans the user query and strictly removes any AI-ranked content that violates such requirements. This ensures adherence to non-negotiable conditions even if semantic ranking finds close matches.

This three-stage pipeline—**Parse → Rank → Filter**—enables domain-agnostic, reliable document analysis within 60 seconds on CPU-only environments.

---

## 📁 Project Structure


/
├── models/               # Contains the pre-downloaded AI model for offline use.
├── input/                # Directory for input PDFs and the persona.json file.
├── output/               # Directory where the final JSON result is written.
├── process_documents.py  # The core Python script with all processing logic.
├── requirements.txt      # A pinned list of all Python dependencies.
├── Dockerfile            # Instructions to build the self-contained Docker image.
└── README.md             # This file.


---

## 🛠️ Technology Stack

- **Language**: Python 3.9
- **Transformer Model**: `sentence-transformers/all-mpnet-base-v2`
- **AI Framework**: PyTorch (CPU-only, offline)
- **PDF Parsing**: PyMuPDF (`fitz`)
- **Containerization**: Docker (slim, offline, <1GB)

---

## ✅ Constraints Compliance

| Requirement               | Status  | Notes |
|--------------------------|---------|-------|
| ⏱️ Execution Time ≤ 60s  | ✅      | Local tests: ~20–30s on 5–10 PDFs |
| 📦 Image Size ≤ 1GB       | ✅      | Final size ~900MB using Python slim & wheel install |
| 🧠 CPU-only Inference     | ✅      | No GPU required |
| 🌐 Offline Execution      | ✅      | No network access; model pre-bundled |

---

## 🚀 Execution Instructions

### Prerequisites

- Docker Desktop installed and running
- Input PDFs and `persona.json` placed in the `input/` folder

### Step 1: Build the Docker Image

From the project root:

```bash
docker build --platform linux/amd64 -t your-solution-name:final .
### Step 1: Build the Docker Image

Navigate to the project's root directory in your PowerShell terminal and run the following command. This will build the Docker image.

```bash
docker build --platform linux/amd64 -t your-solution-name:final .
```


## Step 2: Run the Solution

To process the documents in your input folder, run the following PowerShell command. It will mount your local input and output folders into the container, process the files, and write the challenge1b_output.json file to your output folder.

```bash
docker run --rm -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" --network none your-solution-name:final
```
## Running Different Test Cases

The solution is designed to process any folder named input. To run a different test case (e.g., from a folder named input2):

Rename your current input folder to something else (e.g., input_archive).

Rename the folder you want to test (e.g., input2) to input.

Run the docker run command from Step 2.

The container will automatically process the contents of the newly named input folder.
