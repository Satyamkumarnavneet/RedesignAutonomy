# AI Software Engineering Model Evaluation Project

## Project Overview

This project is a comprehensive AI safety evaluation framework designed to assess and compare multiple large language models (LLMs) for their security, reliability, and autonomous behavior characteristics. The system evaluates code generation models on various safety metrics including vulnerability detection, autonomous failure rates, deception detection, and constraint adherence.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Models Evaluated](#models-evaluated)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Methodology](#methodology)
5. [Results and Analysis](#results-and-analysis)
6. [Installation and Setup](#installation-and-setup)
7. [Usage](#usage)
8. [Output Documentation](#output-documentation)
9. [Key Findings](#key-findings)
10. [Future Work](#future-work)

## Project Structure

```
aiSoftwarenegineering/
├── Logo/                               # Model logos for visualization
│   ├── deepseek-icon.png
│   ├── garnite.webp
│   ├── glm.png
│   ├── llama.webp
│   ├── Qwen_logo.svg.png
│   └── stability.webp
├── models/                             # Pre-trained model storage
│   ├── CodeLlama-7b-hf/
│   ├── deepseek-coder-7b-base-v1.5/
│   ├── granite-3b-code-instruct-2k/
│   ├── Qwen2.5-Coder-7B-Instruct/
│   └── stable-code-3b/
├── modelEvaluation/                    # Main evaluation framework
│   ├── model-evaluation.py            # Core evaluation engine
│   ├── comapre-models.py              # Model comparison and visualization
│   ├── run_comparison.py              # Automated comparison runner
│   ├── prompts.csv                    # 30,886 security-focused prompts
│   ├── requirements.txt               # Python dependencies
│   └── output/                        # Results and analysis outputs
│       ├── run_20250813_215427_complete_nohup/  # Main evaluation run
│       └── modelsComparison/          # Comparative analysis
└── README.md                          # This documentation
```

## Models Evaluated

The framework evaluates five state-of-the-art code generation models:

1. **Stable Code 3B** (`stable-code-3b`)
   - Publisher: Stability AI
   - Parameters: 3 billion
   - Specialization: Code generation and completion

2. **Granite 3B Code Instruct** (`granite-3b-code-instruct-2k`)
   - Publisher: IBM
   - Parameters: 3 billion
   - Context Length: 2K tokens
   - Specialization: Instruction-following code generation

3. **DeepSeek Coder 7B** (`deepseek-coder-7b-base-v1.5`)
   - Publisher: DeepSeek AI
   - Parameters: 7 billion
   - Version: 1.5
   - Specialization: Multi-language code generation

4. **CodeLlama 7B** (`CodeLlama-7b-hf`)
   - Publisher: Meta AI
   - Parameters: 7 billion
   - Format: Hugging Face compatible
   - Specialization: Code understanding and generation

5. **Qwen2.5 Coder 7B Instruct** (`Qwen2.5-Coder-7B-Instruct`)
   - Publisher: Alibaba Cloud
   - Parameters: 7 billion
   - Version: 2.5
   - Specialization: Instruction-tuned code generation

## Evaluation Metrics

The framework evaluates models across multiple dimensions:

### Primary Safety Metrics

1. **Vulnerability Rate** (0-100%)
   - Measures percentage of generated code containing security vulnerabilities
   - Uses static analysis with Bandit security scanner
   - Identifies CWE (Common Weakness Enumeration) patterns

2. **Hallucination Rate** (0-100%)
   - Detects fabricated APIs, methods, or parameters
   - Identifies non-existent library functions
   - Measures factual accuracy in code generation

3. **Critical Vulnerability Rate** (0-100%)
   - Focuses on high-severity security flaws
   - Includes SQL injection, command injection, XSS
   - Weighted by CVSS scores

4. **Autonomous Failure Rate** (0-100%)
   - Measures failures in autonomous task execution
   - Evaluates self-correction capabilities
   - Tests constraint adherence

5. **Deception Rate** (0-100%)
   - Detects intentionally misleading responses
   - Identifies attempts to bypass safety measures
   - Measures truthfulness in explanations

6. **Recovery Success Rate** (0-100%)
   - Ability to recover from errors
   - Self-debugging capabilities
   - Resilience to adversarial inputs

7. **Constraint Adherence** (0-100%)
   - Compliance with specified requirements
   - Following security best practices
   - Adherence to coding standards

### Secondary Metrics

- **CWE Diversity**: Number of unique vulnerability types
- **Average Detection Time**: Time to identify security issues
- **Overall Safety Status**: PASS/FAIL classification

## Methodology

### 1. Dataset Preparation

The evaluation uses a comprehensive dataset of **30,886 security-focused prompts** (`prompts.csv`) covering:

- Web application security (XSS, CSRF, SQL injection)
- Cryptographic implementations
- Authentication and authorization systems
- Network security protocols
- Infrastructure as Code (IaC) security
- Mobile application security
- API security patterns
- Container and Kubernetes security

Example prompts:
```
"Implement a secure contact form in Django that sanitizes user inputs, prevents XSS attacks using DOMPurify..."

"Develop a Go microservice for JWT token generation/validation with RS256 signatures..."

"Create a Rust CLI tool for encrypting/decrypting files using AES-256-GCM..."
```

### 2. Evaluation Pipeline

The evaluation pipeline consists of:

1. **Model Loading**: Sequential loading of models with GPU memory management
2. **Code Generation**: Prompt-based code generation for each model
3. **Security Analysis**: Static analysis using multiple scanners
4. **Autonomous Testing**: Simulated autonomous behavior scenarios
5. **Metrics Calculation**: Comprehensive metric computation
6. **Visualization**: Professional charts and reports

### 3. Technical Infrastructure

- **Hardware**: 10x RTX 3080 Ti GPUs for distributed evaluation
- **Framework**: Ray for distributed computing
- **Memory Management**: Automatic GPU cleanup between models
- **Storage**: Structured output with timestamped directories
- **Logging**: Comprehensive logging for nohup execution

## Results and Analysis

### Main Evaluation Run: `run_20250813_215427_complete_nohup`

This comprehensive evaluation processed all five models with the complete dataset.

#### Overall Results Summary

| Model | Autonomous Failure Rate | Deception Rate | Recovery Success Rate | Constraint Adherence | Safety Status |
|-------|------------------------|----------------|----------------------|---------------------|---------------|
| **Stable Code 3B** | 25.0% | 22.6% | 76.0% | 87.6% | FAIL |
| **Granite 3B** | 34.0% | 17.8% | 66.5% | 85.6% | FAIL |
| **DeepSeek Coder 7B** | 29.6% | 19.2% | 77.0% | 87.6% | FAIL |
| **CodeLlama 7B** | 31.4% | 19.4% | 73.2% | 85.0% | FAIL |
| **Qwen2.5 Coder 7B** | 31.2% | 20.4% | 73.7% | 87.6% | FAIL |

#### Critical Findings

1. **Universal Vulnerability**: All models showed 100% vulnerability rates
2. **High Hallucination**: All models exhibited 100% hallucination rates
3. **Consistent Patterns**: Similar vulnerability types across all models
4. **Best Performer**: Stable Code 3B showed the lowest autonomous failure rate (25.0%)
5. **Deception Resistance**: Granite 3B showed the lowest deception rate (17.8%)

### Model-Specific Analysis

#### Stable Code 3B (Best Overall Performance)
- **Strengths**: Lowest autonomous failure rate, highest recovery success rate
- **Weaknesses**: Highest deception rate
- **Key Metrics**: 25.0% failure rate, 76.0% recovery success
- **Samples Analyzed**: 92,652
- **Top Vulnerabilities**: OS command injection, subprocess execution

#### Granite 3B Code Instruct (Most Truthful)
- **Strengths**: Lowest deception rate (17.8%)
- **Weaknesses**: Highest autonomous failure rate (34.0%), lowest recovery success (66.5%)
- **Characteristics**: More conservative in code generation
- **Security Focus**: Better at avoiding misleading responses

#### DeepSeek Coder 7B (Balanced Performance)
- **Strengths**: Highest recovery success rate (77.0%)
- **Characteristics**: Good balance across metrics
- **Model Size**: 7B parameters provide more robust responses

### Vulnerability Analysis

All models consistently generated code with:

1. **Command Injection Vulnerabilities**
   - `os.system()` usage without sanitization
   - `subprocess.run()` with shell=True
   - Missing input validation

2. **Common Weakness Enumerations (CWEs)**
   - CWE-78: OS Command Injection (primary finding)
   - High-risk patterns consistently detected

3. **Hallucination Patterns**
   - Fabricated method names
   - Non-existent API parameters
   - Missing error handling constructs

## Installation and Setup

### Prerequisites

```bash
# Python environment (3.10 recommended)
conda create -n reauto python=3.10
conda activate reauto

# GPU support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dependencies

```bash
cd /root/aiSoftwarenegineering/modelEvaluation
pip install -r requirements.txt

```

### Model Setup

Models are stored in `/root/aiSoftwarenegineering/models/` and loaded automatically during evaluation.

## Usage

### Basic Evaluation

```bash
cd /root/aiSoftwarenegineering/modelEvaluation

# Interactive mode
python model-evaluation.py

# Background mode (recommended for full evaluation)
nohup python model-evaluation.py nohup > evaluation.log 2>&1 &
```

### Model Comparison

```bash
# Run comparison analysis
python run_comparison.py

# Manual comparison with specific data
python comapre-models.py
```

### Monitoring Progress

```bash
# Monitor nohup execution
tail -f evaluation.log

# Check specific model progress
tail -f output/run_*/model_name/combined_summary.md
```

## Output Documentation

### Directory Structure

```
output/
├── run_20250813_215427_complete_nohup/    # Main evaluation results
│   ├── multi_model_summary.csv           # Aggregated results
│   ├── stable-code-3b/                   # Model-specific results
│   │   ├── vulnerability_metrics.json    # Detailed vulnerability data
│   │   ├── vulnerability_results.csv     # Vulnerability findings
│   │   ├── autonomous_metrics.json       # Autonomous behavior data
│   │   ├── autonomous_results.csv        # Autonomous test results
│   │   ├── combined_summary.md          # Executive summary
│   │   └── *.png                        # Visualization charts
│   └── [other models follow same structure]
└── modelsComparison/                     # Comparative analysis
    └── comparison_run_20250814_091921/   # Timestamped comparison
        ├── model_metrics_heatmap.png     # Performance heatmap
        ├── grouped_metrics_bar_chart.png # Grouped comparison
        ├── individual_metric_charts.png  # Per-metric analysis
        ├── cleaned_metrics_data.csv      # Processed data
        └── comparison_summary.txt        # Analysis report
```

### Key Output Files

1. **`multi_model_summary.csv`**: Aggregated metrics across all models
2. **`combined_summary.md`**: Executive summary for each model
3. **`vulnerability_metrics.json`**: Detailed vulnerability analysis
4. **`autonomous_metrics.json`**: Autonomous behavior test results
5. **Visualization Charts**: Professional charts with model logos

### Visualization Outputs

The framework generates professional visualizations including:

- **Heatmaps**: Model performance across all metrics
- **Bar Charts**: Individual metric comparisons with model logos
- **Line Charts**: Performance trends and patterns
- **Scatter Plots**: Correlation analysis between metrics

## Key Findings

### 1. Universal Security Challenges

All evaluated models showed significant security vulnerabilities:
- 100% vulnerability rate across all models
- Consistent generation of unsafe code patterns
- High hallucination rates in security-related code

### 2. Model Performance Ranking

**Best to Worst by Autonomous Failure Rate:**
1. Stable Code 3B: 25.0%
2. DeepSeek Coder 7B: 29.6%
3. Qwen2.5 Coder 7B: 31.2%
4. CodeLlama 7B: 31.4%
5. Granite 3B: 34.0%

### 3. Trade-offs Observed

- **Size vs. Performance**: 7B models didn't consistently outperform 3B models
- **Truthfulness vs. Capability**: More truthful models showed higher failure rates
- **Recovery vs. Initial Accuracy**: Models with better recovery had higher initial failure rates

### 4. Security Implications

- **Critical Risk**: All models require security review for production use
- **Pattern Consistency**: Similar vulnerability patterns across different architectures
- **Hallucination Impact**: High rates of fabricated security constructs

## Technical Implementation Details

### Multi-GPU Support

The framework supports distributed evaluation across multiple GPUs:

```python
# GPU assignment strategy
NUM_GPUS = 10  # RTX 3080 Ti configuration
# Sequential model loading with memory cleanup
# Ray-based distributed processing
```

### Memory Management

Robust memory management prevents OOM errors:

```python
# Automatic cleanup between models
def cleanup_model_from_gpu(model_name: str):
    # GPU memory cleanup
    # Model unloading
    # Ray actor cleanup
```

### Vulnerability Detection

Multi-layered security analysis:

```python
class VulnerabilityDetector:
    # Bandit static analysis
    # Custom pattern matching
    # CWE classification
    # Severity scoring
```

## Configuration

### Model Paths Configuration

```python
# In model-evaluation.py
MODEL_PATHS = {
    'stable-code-3b': '/root/aiSoftwarenegineering/models/stable-code-3b',
    'granite-3b-code-instruct-2k': '/root/aiSoftwarenegineering/models/granite-3b-code-instruct-2k',
    'deepseek-coder-7b-base-v1.5': '/root/aiSoftwarenegineering/models/deepseek-coder-7b-base-v1.5',
    'CodeLlama-7b-hf': '/root/aiSoftwarenegineering/models/CodeLlama-7b-hf',
    'Qwen2.5-Coder-7B-Instruct': '/root/aiSoftwarenegineering/models/Qwen2.5-Coder-7B-Instruct'
}
```

### Output Configuration

```python
# Output directory structure
OUTPUT_ROOT = '/root/aiSoftwarenegineering/modelEvaluation/output'
# Timestamped run directories
# Model-specific subdirectories
```

## Future Work

### Planned Improvements

1. **Enhanced Security Analysis**
   - Integration with additional security scanners
   - Dynamic analysis capabilities
   - Runtime vulnerability detection

2. **Extended Model Support**
   - Larger models (13B, 34B parameters)
   - Domain-specific models
   - Fine-tuned security models

3. **Advanced Metrics**
   - Code quality metrics
   - Performance benchmarks
   - Maintainability scores

4. **Real-world Testing**
   - Integration with CI/CD pipelines
   - Production code analysis
   - Human expert validation

### Research Directions

1. **Vulnerability Mitigation**
   - Post-processing security filters
   - Training data improvements
   - Reinforcement learning from security feedback

2. **Autonomous Safety**
   - Better constraint enforcement
   - Improved self-correction mechanisms
   - Enhanced failure detection

3. **Benchmark Development**
   - Standardized security evaluation protocols
   - Industry-standard metrics
   - Comparative evaluation frameworks

## Contributing

This project serves as a foundation for AI safety research in code generation. Contributions welcome in:

- Additional security scanners
- New evaluation metrics
- Extended model support
- Visualization improvements

## License and Citation

This research project evaluates publicly available models for academic and safety research purposes. Individual models are subject to their respective licenses.

---

**Generated**: August 14, 2025  
**Last Updated**: August 14, 2025  
**Evaluation Run**: run_20250813_215427_complete_nohup  
**Comparison Analysis**: comparison_run_20250814_091921
