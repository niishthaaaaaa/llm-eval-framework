"""
LLM Evaluation Framework
=========================
Tests multiple prompt templates, records outputs, scores them with
evaluation metrics, and generates experiment reports.

Usage:
    python eval_framework.py

Requirements:
    pip install anthropic rich pandas jinja2
"""

import json
import time
import re
import statistics
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import anthropic


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class PromptTemplate:
    name: str
    template: str          # Use {input} as placeholder for the test input
    description: str = ""

    def render(self, input_text: str) -> str:
        return self.template.replace("{input}", input_text)


@dataclass
class TestCase:
    id: str
    input: str
    expected_keywords: list[str] = field(default_factory=list)   # Words that should appear
    forbidden_keywords: list[str] = field(default_factory=list)  # Words that must NOT appear
    min_length: int = 10
    max_length: int = 2000


@dataclass
class EvalResult:
    template_name: str
    test_case_id: str
    prompt_used: str
    raw_output: str
    latency_ms: float
    scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    error: str | None = None


@dataclass
class ExperimentReport:
    experiment_id: str
    timestamp: str
    model: str
    templates: list[str]
    test_cases: int
    results: list[EvalResult]
    summary: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Scoring metrics
# ─────────────────────────────────────────────

class Scorer:
    """Collection of evaluation metrics."""

    @staticmethod
    def keyword_coverage(output: str, keywords: list[str]) -> float:
        """Fraction of expected keywords present (case-insensitive)."""
        if not keywords:
            return 1.0
        output_lower = output.lower()
        hits = sum(1 for kw in keywords if kw.lower() in output_lower)
        return hits / len(keywords)

    @staticmethod
    def forbidden_penalty(output: str, forbidden: list[str]) -> float:
        """1.0 if none forbidden words appear, else 0.0."""
        if not forbidden:
            return 1.0
        output_lower = output.lower()
        violations = sum(1 for kw in forbidden if kw.lower() in output_lower)
        return 0.0 if violations > 0 else 1.0

    @staticmethod
    def length_score(output: str, min_len: int, max_len: int) -> float:
        """1.0 if within bounds, decays outside."""
        n = len(output)
        if min_len <= n <= max_len:
            return 1.0
        if n < min_len:
            return n / min_len
        # Over max: soft penalty
        return max(0.0, 1.0 - (n - max_len) / max_len)

    @staticmethod
    def coherence_heuristic(output: str) -> float:
        """Simple heuristic: penalise repetition and very short sentences."""
        if not output.strip():
            return 0.0
        sentences = [s.strip() for s in re.split(r'[.!?]', output) if s.strip()]
        if not sentences:
            return 0.5
        # Penalise duplicate sentences
        unique_ratio = len(set(sentences)) / len(sentences)
        # Reward reasonable sentence length (5–30 words)
        avg_words = statistics.mean(len(s.split()) for s in sentences)
        length_ok = 1.0 if 5 <= avg_words <= 30 else max(0.3, 1 - abs(avg_words - 17) / 30)
        return round((unique_ratio + length_ok) / 2, 3)

    def score(self, result: EvalResult, test_case: TestCase) -> EvalResult:
        """Compute all metrics and attach to result."""
        out = result.raw_output
        scores = {
            "keyword_coverage": self.keyword_coverage(out, test_case.expected_keywords),
            "forbidden_penalty": self.forbidden_penalty(out, test_case.forbidden_keywords),
            "length_score":      self.length_score(out, test_case.min_length, test_case.max_length),
            "coherence":         self.coherence_heuristic(out),
        }
        result.scores = scores
        result.overall_score = round(statistics.mean(scores.values()), 4)
        return result


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

class EvalRunner:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 512):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.scorer = Scorer()

    def run_single(self, template: PromptTemplate, test_case: TestCase) -> EvalResult:
        prompt = template.render(test_case.input)
        result = EvalResult(
            template_name=template.name,
            test_case_id=test_case.id,
            prompt_used=prompt,
            raw_output="",
            latency_ms=0.0,
        )
        try:
            t0 = time.perf_counter()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            result.raw_output = response.content[0].text
        except Exception as exc:
            result.error = str(exc)
            result.latency_ms = 0.0

        return self.scorer.score(result, test_case)

    def run_experiment(
        self,
        templates: list[PromptTemplate],
        test_cases: list[TestCase],
        experiment_id: str | None = None,
    ) -> ExperimentReport:
        exp_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n🔬  Experiment: {exp_id}  |  model: {self.model}")
        print(f"    Templates : {[t.name for t in templates]}")
        print(f"    Test cases: {len(test_cases)}\n")

        all_results: list[EvalResult] = []
        total = len(templates) * len(test_cases)
        idx = 0

        for tc in test_cases:
            for tmpl in templates:
                idx += 1
                print(f"  [{idx:>3}/{total}] template={tmpl.name:<20} case={tc.id}", end=" ... ")
                res = self.run_single(tmpl, tc)
                all_results.append(res)
                status = f"✓  score={res.overall_score:.3f}  latency={res.latency_ms:.0f}ms"
                if res.error:
                    status = f"✗  ERROR: {res.error}"
                print(status)

        report = ExperimentReport(
            experiment_id=exp_id,
            timestamp=datetime.now().isoformat(),
            model=self.model,
            templates=[t.name for t in templates],
            test_cases=len(test_cases),
            results=all_results,
        )
        report.summary = _compute_summary(report)
        return report


# ─────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────

def _compute_summary(report: ExperimentReport) -> dict:
    from collections import defaultdict
    by_template: dict[str, list[float]] = defaultdict(list)
    by_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in report.results:
        if r.error:
            continue
        by_template[r.template_name].append(r.overall_score)
        for metric, val in r.scores.items():
            by_metric[r.template_name][metric].append(val)

    template_summary = {}
    for tmpl, scores in by_template.items():
        template_summary[tmpl] = {
            "mean_overall": round(statistics.mean(scores), 4),
            "min_overall":  round(min(scores), 4),
            "max_overall":  round(max(scores), 4),
            "metrics": {
                metric: round(statistics.mean(vals), 4)
                for metric, vals in by_metric[tmpl].items()
            },
        }

    best = max(template_summary, key=lambda t: template_summary[t]["mean_overall"]) if template_summary else "N/A"
    latencies = [r.latency_ms for r in report.results if not r.error]

    return {
        "best_template": best,
        "template_summary": template_summary,
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "total_runs": len(report.results),
        "error_count": sum(1 for r in report.results if r.error),
    }


# ─────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────

def save_json_report(report: ExperimentReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.experiment_id}_report.json"
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    return path


def save_html_report(report: ExperimentReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.experiment_id}_report.html"

    summary = report.summary
    ts = summary.get("template_summary", {})

    # Build template rows
    tmpl_rows = ""
    for tmpl, data in ts.items():
        metrics_html = "".join(
            f"<td>{v:.3f}</td>" for v in data["metrics"].values()
        )
        best_class = ' class="best"' if tmpl == summary.get("best_template") else ""
        tmpl_rows += (
            f'<tr{best_class}>'
            f'<td><strong>{tmpl}</strong></td>'
            f'<td>{data["mean_overall"]:.4f}</td>'
            f'<td>{data["min_overall"]:.4f}</td>'
            f'<td>{data["max_overall"]:.4f}</td>'
            f'{metrics_html}'
            f'</tr>\n'
        )

    # Build results rows
    result_rows = ""
    for r in report.results:
        score_cells = "".join(f'<td>{v:.3f}</td>' for v in r.scores.values())
        preview = r.raw_output[:120].replace("<", "&lt;").replace(">", "&gt;")
        err_cell = f'<td class="error">{r.error}</td>' if r.error else "<td>—</td>"
        result_rows += (
            f"<tr>"
            f"<td>{r.template_name}</td>"
            f"<td>{r.test_case_id}</td>"
            f"<td>{r.overall_score:.4f}</td>"
            f"{score_cells}"
            f"<td>{r.latency_ms:.0f}</td>"
            f'<td class="preview">{preview}…</td>'
            f"{err_cell}"
            f"</tr>\n"
        )

    metric_headers = ""
    if ts:
        first = next(iter(ts.values()))
        for m in first["metrics"]:
            metric_headers += f"<th>{m}</th>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Eval – {report.experiment_id}</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d2e; --accent: #7c6af7;
    --green: #4ade80; --red: #f87171; --text: #e2e8f0; --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: .25rem; }}
  h2 {{ font-size: 1.1rem; color: var(--accent); margin: 2rem 0 .75rem; text-transform: uppercase; letter-spacing: .05em; }}
  .meta {{ color: var(--muted); font-size: .85rem; margin-bottom: 2rem; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .kpi {{ background: var(--surface); border-radius: 10px; padding: 1.2rem; }}
  .kpi .label {{ font-size: .75rem; color: var(--muted); margin-bottom: .3rem; }}
  .kpi .value {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; background: var(--surface); border-radius: 10px; overflow: hidden; }}
  th {{ background: #252840; padding: .6rem .8rem; text-align: left; color: var(--muted); font-weight: 600; }}
  td {{ padding: .55rem .8rem; border-top: 1px solid #252840; vertical-align: top; }}
  tr.best td {{ background: #1e2d1e; }}
  tr:hover td {{ background: #20233a; }}
  .preview {{ color: var(--muted); max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .error {{ color: var(--red); }}
  .badge {{ display: inline-block; padding: .15rem .5rem; border-radius: 4px; font-size: .75rem; font-weight: 700;
            background: #2a2060; color: var(--accent); margin-left: .4rem; }}
</style>
</head>
<body>
<h1>🔬 LLM Evaluation Report <span class="badge">{report.experiment_id}</span></h1>
<p class="meta">
  Model: <strong>{report.model}</strong> &nbsp;|&nbsp;
  Timestamp: {report.timestamp} &nbsp;|&nbsp;
  Templates tested: {', '.join(report.templates)}
</p>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Best Template</div><div class="value" style="font-size:1.1rem">{summary.get('best_template','—')}</div></div>
  <div class="kpi"><div class="label">Total Runs</div><div class="value">{summary.get('total_runs',0)}</div></div>
  <div class="kpi"><div class="label">Errors</div><div class="value" style="color:var(--red)">{summary.get('error_count',0)}</div></div>
  <div class="kpi"><div class="label">Avg Latency</div><div class="value">{summary.get('avg_latency_ms',0)}<span style="font-size:.8rem;color:var(--muted)"> ms</span></div></div>
</div>

<h2>Template Summary</h2>
<table>
  <thead><tr><th>Template</th><th>Mean Score</th><th>Min</th><th>Max</th>{metric_headers}</tr></thead>
  <tbody>{tmpl_rows}</tbody>
</table>

<h2>All Results</h2>
<table>
  <thead>
    <tr>
      <th>Template</th><th>Case</th><th>Overall</th>
      <th>Keyword Coverage</th><th>Forbidden Penalty</th><th>Length</th><th>Coherence</th>
      <th>Latency (ms)</th><th>Output Preview</th><th>Error</th>
    </tr>
  </thead>
  <tbody>{result_rows}</tbody>
</table>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


def save_markdown_report(report: ExperimentReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.experiment_id}_report.md"
    s = report.summary
    ts = s.get("template_summary", {})

    lines = [
        f"# LLM Evaluation Report — `{report.experiment_id}`",
        "",
        f"- **Model**: `{report.model}`",
        f"- **Timestamp**: {report.timestamp}",
        f"- **Templates**: {', '.join(report.templates)}",
        f"- **Test cases**: {report.test_cases}",
        f"- **Best template**: **{s.get('best_template', 'N/A')}**",
        f"- **Avg latency**: {s.get('avg_latency_ms', 0)} ms",
        "",
        "## Template Summary",
        "",
        "| Template | Mean Score | Min | Max | Keyword Coverage | Forbidden Penalty | Length | Coherence |",
        "|----------|-----------|-----|-----|-----------------|-------------------|--------|-----------|",
    ]
    for tmpl, data in ts.items():
        m = data["metrics"]
        marker = " ⭐" if tmpl == s.get("best_template") else ""
        lines.append(
            f"| {tmpl}{marker} | {data['mean_overall']:.4f} | {data['min_overall']:.4f} | {data['max_overall']:.4f}"
            f" | {m.get('keyword_coverage',0):.3f} | {m.get('forbidden_penalty',0):.3f}"
            f" | {m.get('length_score',0):.3f} | {m.get('coherence',0):.3f} |"
        )

    lines += ["", "## Detailed Results", ""]
    for r in report.results:
        lines.append(f"### {r.template_name} × {r.test_case_id}")
        lines.append(f"- **Overall score**: `{r.overall_score}`  |  **Latency**: {r.latency_ms} ms")
        if r.error:
            lines.append(f"- ❌ Error: {r.error}")
        else:
            lines.append(f"- Scores: {r.scores}")
            preview = r.raw_output[:200].replace("\n", " ")
            lines.append(f"- Output preview: *{preview}…*")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ─────────────────────────────────────────────
# Demo experiment
# ─────────────────────────────────────────────

def run_demo():
    # ── Prompt templates to compare ──────────────────────────────────────
    templates = [
        PromptTemplate(
            name="direct",
            template="{input}",
            description="Bare input with no system framing",
        ),
        PromptTemplate(
            name="expert_framing",
            template=(
                "You are a senior expert. Provide a concise, accurate, and well-structured "
                "answer to the following question:\n\n{input}"
            ),
            description="Expert persona added before the question",
        ),
        PromptTemplate(
            name="step_by_step",
            template=(
                "Think step by step and explain your reasoning clearly before giving a final answer.\n\n"
                "Question: {input}"
            ),
            description="Chain-of-thought prompting",
        ),
        PromptTemplate(
            name="structured_output",
            template=(
                "Answer the question below. Format your response with:\n"
                "1. A one-sentence summary\n"
                "2. Key points (bullet list)\n"
                "3. A brief conclusion\n\n"
                "Question: {input}"
            ),
            description="Structured output instructions",
        ),
    ]

    # ── Test cases ────────────────────────────────────────────────────────
    test_cases = [
        TestCase(
            id="tc_photosynthesis",
            input="What is photosynthesis and why is it important?",
            expected_keywords=["sunlight", "carbon dioxide", "oxygen", "glucose", "plants"],
            forbidden_keywords=["I don't know", "I cannot"],
            min_length=80,
            max_length=800,
        ),
        TestCase(
            id="tc_recursion",
            input="Explain recursion in programming with a simple example.",
            expected_keywords=["function", "base case", "call", "stack"],
            forbidden_keywords=["I don't know"],
            min_length=100,
            max_length=900,
        ),
        TestCase(
            id="tc_climate_change",
            input="What are the main causes and effects of climate change?",
            expected_keywords=["greenhouse", "carbon", "temperature", "emissions"],
            forbidden_keywords=[],
            min_length=120,
            max_length=1000,
        ),
    ]

    runner = EvalRunner(model="claude-sonnet-4-20250514", max_tokens=600)
    report = runner.run_experiment(templates, test_cases, experiment_id="demo_experiment")

    # ── Save outputs ──────────────────────────────────────────────────────
    out = Path("eval_reports")
    json_path = save_json_report(report, out)
    html_path = save_html_report(report, out)
    md_path   = save_markdown_report(report, out)

    print("\n" + "─" * 60)
    print("📊  Reports saved:")
    print(f"    JSON : {json_path}")
    print(f"    HTML : {html_path}")
    print(f"    MD   : {md_path}")
    print("─" * 60)

    # ── Print quick summary ───────────────────────────────────────────────
    s = report.summary
    print(f"\n🏆  Best template : {s['best_template']}")
    print(f"⏱   Avg latency   : {s['avg_latency_ms']} ms")
    print(f"❌  Errors         : {s['error_count']}\n")
    print("Template scores:")
    for tmpl, data in s["template_summary"].items():
        star = " ⭐" if tmpl == s["best_template"] else "   "
        print(f"  {star} {tmpl:<25} mean={data['mean_overall']:.4f}")
    print()
    return report


if __name__ == "__main__":
    run_demo()
