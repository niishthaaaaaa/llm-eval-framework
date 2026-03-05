![LLM Evaluation Framework]

A lightweight framework for evaluating prompt strategies across Large Language Model (LLM) tasks using automated scoring metrics and experiment reports.




## Evaluation Dashboard

![LLM Evaluation Report](dashboard.png)




\## Features


\- Compare multiple prompt strategies

\- Run experiments across multiple test cases

\- Automated evaluation metrics:

  - Keyword coverage

  - Forbidden word detection

  - Response length scoring

  - Coherence heuristic

\- Generates experiment reports in:

  - HTML dashboard

  - Markdown

  - JSON
---


\## Project Structure


llm-eval-framework

│

├── eval\_framework.py

├── eval\_reports/

│   ├── demo\_experiment\_report.html

│   ├── demo\_experiment\_report.md

│   └── demo\_experiment\_report.json




---
\## Tech Stack


\- Python

\- Anthropic API

\- Prompt Engineering

\- LLM Evaluation



\## Running the Framework


Requires Python 3.10+


Install dependencies: 
```bash
pip install anthropic
```




\### Set your API key


On Linux/Mac: export ANTHROPIC\_API\_KEY="your-api-key"

On Windows PowerShell: $env:ANTHROPIC\_API\_KEY="your-api-key"



\### Run the evaluation: python eval\_framework.py


Reports will be generated inside: eval\_reports/
These include:

\- HTML dashboard

\- Markdown report

\- JSON experiment data



---
After running the framework, an HTML dashboard is generated showing:


\- prompt template performance

\- evaluation metrics

\- experiment summaries



---
\## Author

Nishtha Sharma
 

\## License
MIT






