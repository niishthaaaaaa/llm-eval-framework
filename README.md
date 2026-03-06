# LLM Evaluation Framework

A lightweight framework for evaluating prompt strategies across Large Language Model (LLM) tasks using automated scoring metrics and experiment reports.

> Useful for comparing prompt engineering strategies before deploying to production.

---

## Evaluation Dashboard

![LLM Evaluation Report](dashboard.png)

---

## Features

- Compare multiple prompt strategies
- Run experiments across multiple test cases
- Automated evaluation metrics:
  - Keyword coverage
  - Forbidden word detection
  - Response length scoring
  - Coherence heuristic
- Generates experiment reports in:
  - HTML dashboard
  - Markdown
  - JSON

---

## Project Structure

```
llm-eval-framework/
│
├── eval_framework.py
├── README.md
├── requirements.txt
├── .gitignore
└── eval_reports/          ← auto-created on first run
    ├── demo_experiment_report.html
    ├── demo_experiment_report.md
    └── demo_experiment_report.json
```

---

## Tech Stack

- Python 3.10+
- Google Gemini API (free tier)
- Prompt Engineering
- LLM Evaluation

---

## Running the Framework

Requires Python 3.10+

### 1. Install dependencies

```bash
pip install google-generativeai
```

### 2. Set your API key

On Linux/Mac:
```bash
export GEMINI_API_KEY="your-api-key"
```

On Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="your-api-key"
```

### 3. Run the evaluation

```bash
python eval_framework.py
```

Reports will be generated inside `eval_reports/` and include:

- HTML dashboard
- Markdown report
- JSON experiment data

---

## Output

After running the framework, an HTML dashboard is generated showing:

- Prompt template performance
- Evaluation metrics
- Experiment summaries

---

## Author

Nishtha Sharma

## License

MIT
