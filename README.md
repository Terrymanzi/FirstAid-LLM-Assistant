# FirstAidQA – LLM-Based First Aid Question Answering Assistant

## Project Overview

This project presents a domain-specific generative Question Answering (QA) assistant trained on first aid and emergency response knowledge. The assistant is designed to provide clear, step-by-step emergency guidance using a fine-tuned Large Language Model (LLM).

The chatbot is built using the FirstAidQA dataset, a high-quality synthetic dataset containing 5,500 validated question-answer pairs derived from the Vital First Aid Book (2019).

## Purpose and Motivation

Emergency situations require fast, accurate, and structured guidance. Many individuals lack immediate access to professional medical support. This assistant aims to:

- Provide structured first-aid instructions
- Deliver clear step-by-step emergency responses
- Demonstrate how domain-specific fine-tuning improves LLM reliability
- Explore training strategies for safety-sensitive applications

Disclaimer: This chatbot is for educational purposes only and does not replace professional medical advice.

## Dataset

Dataset: i-am-mushfiq/FirstAidQA
Source: Hugging Face
Size: Approximately 5,500 question-answer pairs
Domain: First aid and emergency response

Dataset characteristics:

- Step-by-step emergency procedures
- Casualty movement guidance
- Environmental risk considerations
- Human-validated for safety and accuracy
- Structured QA format

## Model Architecture and Experiments

This project uses **TinyLlama-1.1B-Chat-v1.0** with Parameter-Efficient Fine-Tuning (PEFT) using LoRA.

All experiments use:

- **4-bit quantization** (bitsandbytes) for memory efficiency
- **LoRA** (Low-Rank Adaptation) for parameter-efficient training
- **Target modules**: q_proj, v_proj
- **Batch size**: 2 with gradient accumulation steps of 8 (effective batch size: 16)
- **Max sequence length**: 512 tokens

Three experimental configurations:

| Experiment | Learning Rate | LoRA Rank | Epochs | Focus                          |
| ---------- | ------------- | --------- | ------ | ------------------------------ |
| Exp 1      | 2e-4          | 16        | 2      | Higher LR, moderate rank       |
| Exp 2      | 5e-5          | 16        | 3      | Lower LR, more training        |
| Exp 3      | 2e-4          | 32        | 1      | Higher rank, quick convergence |

Each experiment:

- Loads a fresh base model (prevents LoRA stacking)
- Trains independently with different hyperparameters
- Saves adapter weights separately
- Records training time, validation loss, and perplexity

Each experiment is compared using comprehensive validation metrics.

## Preprocessing Pipeline

The dataset undergoes structured preprocessing:

- **Train-validation split**: 90/10 with seed=42 for reproducibility

- **Instruction-response formatting**:

  ```
  ### Instruction:
  You are a professional first aid assistant.
  Provide clear, step-by-step emergency guidance.
  This is for educational purposes only.

  Question:
  {question}

  ### Response:
  {answer}
  ```

- **Tokenization**: Using TinyLlama tokenizer (SentencePiece-based)

- **Truncation and padding**: Max length 512 tokens

- **Format conversion**: PyTorch tensors for training

- **Random seeds**: Set to 42 for full reproducibility

## Evaluation Metrics

The chatbot is evaluated using multiple NLP metrics:

- **BLEU Score** – Measures n-gram overlap between generated and reference answers
- **ROUGE Scores** (ROUGE-1, ROUGE-2, ROUGE-L) – Recall-oriented evaluation
- **Perplexity** – Language model confidence (exp(eval_loss))
- **Qualitative Testing**:
  - 5 in-domain first-aid scenarios (CPR, burns, bleeding, choking, fractures)
  - 2 out-of-domain questions (geography, cooking)
- **Base vs Fine-tuned Comparison** – Direct quality assessment

Evaluation conducted on:

- 50 validation samples for automated metrics
- Best performing experiment selected by lowest validation loss
- Fine-tuned model shows significant improvements in domain-specific responses

## Results Summary

Results are captured in `experiment_results.csv` after training completion. Each experiment records:

- Validation loss
- Training time (seconds)
- Perplexity (exp(eval_loss))
- BLEU and ROUGE scores on 50 validation samples

**Key Findings:**

- Fine-tuned models provide clear, structured first-aid instructions
- Responses align with emergency guidance protocols
- Out-of-domain questions handled gracefully
- Best model automatically selected based on lowest validation loss
- LoRA enables efficient training: <1% trainable parameters
- All experiments trainable on Google Colab T4 GPU (free tier)

**Sample Output Quality:**

- Base model: Generic or incomplete responses
- Fine-tuned model: Step-by-step emergency instructions with medical context

## User Interface

An interactive web interface is deployed using **Gradio** with:

- **Input**: Multi-line text box for questions
- **Output**: Formatted response from fine-tuned model
- **Examples**: Pre-loaded first-aid scenarios
- **Safety Disclaimer**: Educational use warning prominently displayed
- **Share Link**: Public URL for testing (share=True)

**Features:**

- Real-time generation using best fine-tuned adapter
- Temperature sampling (0.7) for natural responses
- Response extraction (removes prompt echo)
- Clean, user-friendly interface

**Example Interaction:**

```
User: What should I do if someone is choking?

Assistant:
1. Assess if the person can cough or speak
2. If unable to breathe, perform 5 back blows between shoulder blades
3. Follow with 5 abdominal thrusts (Heimlich maneuver)
4. Alternate back blows and abdominal thrusts until object is dislodged
5. Call emergency services (911) if obstruction persists
6. Begin CPR if person becomes unconscious
```

## Project Structure

```
FirstAid-LLM-Assistant/
├── notebooks/
│   └── FirstAid_LLM_Assistant.ipynb    # Main training and evaluation notebook
├── experiments/                         # Placeholder for experiment artifacts
├── models/                              # Placeholder for saved models
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
└── .gitignore                           # Git ignore rules
```

**Generated during execution:**

```
exp1/, exp2/, exp3/                     # Training checkpoints and adapters
exp1_adapter/, exp2_adapter/, exp3_adapter/  # LoRA adapter weights
final_firstaid_model/                   # Best model with adapter
experiment_results.csv                  # Metrics comparison table
```

## How to Run

### Google Colab (Recommended)

1. **Upload notebook** to Google Colab or open directly from GitHub
2. **Set GPU runtime**:
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4)
3. **Run all cells sequentially** (Runtime → Run all)
   - Cell 1-3: Setup and imports
   - Cell 4-9: Dataset loading and preprocessing
   - Cell 10-17: Tokenization and model configuration
   - Cell 18-23: Run 3 experiments (takes ~2-4 hours total)
   - Cell 24-26: Identify and load best model
   - Cell 27-35: Evaluation (BLEU, ROUGE, qualitative tests)
   - Cell 36-37: Gradio deployment
   - Cell 38: Save final model
   - Cell 39-44: Learning curves visualization

4. **Access Gradio interface** via the generated public link

### Local Setup (Requires GPU)

```bash
git clone https://github.com/Terrymanzi/FirstAid-LLM-Assistant.git
cd FirstAid-LLM-Assistant
pip install -r requirements.txt
jupyter notebook notebooks/FirstAid_LLM_Assistant.ipynb
```

**Requirements:**

- CUDA-capable GPU (≥8GB VRAM)
- Python 3.8+
- CUDA 11.8+ with compatible PyTorch

## Code Quality

- **Modular design**: Reusable `run_experiment()` function
- **Reproducibility**: Fixed random seeds (SEED=42)
- **Clean structure**: Logical cell organization
- **No model reuse bug**: Each experiment loads fresh base model
- **Comprehensive logging**: Progress indicators and metrics
- **Error handling**: GPU availability checks
- **Documentation**: Explanatory markdown cells throughout
- **Best practices**: Independent experiments, proper adapter saving
- **Memory efficient**: 4-bit quantization + LoRA
- **Clear outputs**: Formatted tables and comparison sections

## Technical Highlights

- **Parameter Efficiency**: LoRA reduces trainable params to <1% of total
- **Memory Optimization**: 4-bit quantization enables T4 GPU training
- **Experiment Independence**: Fresh model loading prevents adapter stacking
- **Automatic Selection**: Best model chosen by validation loss
- **Response Extraction**: Clean generation without prompt leakage
- **Safety Integration**: Medical disclaimers in prompt and interface

## Future Improvements

- Expand dataset with more emergency scenarios
- Add multi-lingual support
- Implement retrieval-augmented generation (RAG)
- Fine-tune larger models (Llama 2 7B, Mistral 7B)
- Deploy as web service or mobile app
- Add citation/source attribution for medical claims

## Citation

If you use this project, please cite:

```bibtex
@misc{firstaid_llm_assistant_2026,
  title={FirstAid-LLM-Assistant: Domain-Specific Generative QA with LoRA},
  author={Terry Manzi},
  year={2026},
  institution={African Leadership University},
  course={Machine Learning Techniques 1 - Summative Assessment}
}
```

## License

This project is for educational purposes. The FirstAidQA dataset and TinyLlama model have their respective licenses.

## Acknowledgments

- **Dataset**: i-am-mushfiq/FirstAidQA on Hugging Face
- **Base Model**: TinyLlama-1.1B-Chat-v1.0 by TinyLlama team
- **PEFT Library**: Hugging Face PEFT for LoRA implementation
- **Institution**: African Leadership University
