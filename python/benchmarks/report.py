"""Create a transparent Markdown report from benchmark summary JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.data import BENCHMARK_ROOT


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def generate_report(
    summary_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    summary_path = summary_path or BENCHMARK_ROOT / "results" / "reference" / "summary.json"
    output_path = output_path or BENCHMARK_ROOT.parent.parent / "docs" / "benchmark-report.md"
    if not summary_path.exists():
        raise FileNotFoundError(f"benchmark summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    thresholds = summary.get("metric_thresholds", {})
    lines = [
        "# AgentKnowledgeHub 量化评测报告",
        "",
        "> 本报告由 `akh-benchmark report` 根据原始 JSONL 自动生成，不包含 API Key 或私有 Base URL。",
        "",
    ]
    if not summary.get("formal_complete", False):
        lines.extend(
            [
                "> **部分运行：当前样本数未覆盖正式验收规模。未完成套件的指标仅用于冒烟诊断，",
                "> 不判定 PASS/FAIL，也不能作为正式基准结论。**",
                "",
            ]
        )
    lines.extend(
        [
            "## 运行配置",
            "",
            f"- 运行类型：`{summary.get('run_kind', 'unknown')}`",
            f"- 数据版本：`{summary.get('data_version', 'unknown')}`",
            f"- 生成模型：`{summary.get('config', {}).get('llm_model', 'unknown')}`",
            f"- Embedding 模型：`{summary.get('config', {}).get('embedding_model', 'unknown')}`",
            f"- Chunk：`{summary.get('config', {}).get('chunk_size_tokens', 'unknown')}` / overlap "
            f"`{summary.get('config', {}).get('chunk_overlap_tokens', 'unknown')}`",
            f"- Top-K：`{summary.get('config', {}).get('top_k', 'unknown')}`；temperature：`0`；seed：`42`",
            f"- 套件完整性：`{json.dumps(summary.get('suite_completeness', {}), ensure_ascii=False)}`",
            "",
            "## 实际样本数",
            "",
            f"```json\n{json.dumps(summary.get('sample_counts', {}), ensure_ascii=False, indent=2)}\n```",
            "",
            "## 指标与验收",
            "",
            "| 指标 | 实测值 | 门槛 | 结果 |",
            "|---|---:|---:|:---:|",
        ]
    )
    for name in sorted(metrics):
        value = metrics.get(name)
        threshold = thresholds.get(name)
        passed = summary.get("gates", {}).get(name)
        status = "PASS" if passed is True else "FAIL" if passed is False else "N/A"
        lines.append(f"| `{name}` | {_format_value(value)} | {_format_value(threshold)} | {status} |")
    confidence = summary.get("confidence_intervals", {})
    if confidence:
        lines.extend(["", "## 95% 置信区间", ""])
        if not summary.get("formal_complete", False):
            lines.append("以下区间来自部分样本，仅用于验证计算流程。")
            lines.append("")
        for name, interval in confidence.items():
            lines.append(
                f"- `{name}`：{_format_value(interval.get('estimate'))} "
                f"[{_format_value(interval.get('lower'))}, {_format_value(interval.get('upper'))}]"
            )
    vision_status = "enabled" if summary.get("config", {}).get("vision_enabled") else "disabled (OCR only)"
    lines.extend(
        [
            "",
            "## 调用、成本与延迟",
            "",
            f"```json\n{json.dumps(summary.get('usage', {}), ensure_ascii=False, indent=2)}\n```",
            f"- 用量范围：{summary.get('usage_scope', 'unknown')}",
            "",
            "## 错误与限制",
            "",
        ]
    )
    errors = summary.get("errors") or []
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- 本次运行未记录额外错误。")
    lines.extend(
        [
            "",
            f"- 视觉描述：{vision_status}",
            "- 所有未达门槛的项目保留原始结果，不通过调整样本来修饰指标。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
