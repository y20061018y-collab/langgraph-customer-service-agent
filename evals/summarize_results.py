"""
把 evals/results/*.json 汇总成"全部运行记录"表格（时间、通过数、失败用例）。

用法（项目根目录）：
    python evals/summarize_results.py                              # 打印表格
    python evals/summarize_results.py --append docs/tuning-log.md  # 追加到文档末尾

表格完全由结果文件生成，不手工编辑。--append 只在目标文件末尾追加一个小节，
不改动原有内容；目标文件已含本小节标记时拒绝重复追加。
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MARKER = "<!-- generated-by: evals/summarize_results.py -->"
HEADING = "## 附录：全部运行记录（脚本生成）"


def load_runs() -> list[dict]:
    runs = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data["meta"]
        results = data["results"]
        runs.append(
            {
                "file": os.path.basename(path),
                "time": datetime.strptime(meta["time"], "%Y%m%d_%H%M%S"),
                "model": meta.get("model") or "未记录",
                "cases_file": meta.get("cases_file") or "未记录",
                "passed": meta["passed"],
                "total": meta["total"],
                "failed_ids": [r["id"] for r in results if not r["passed"]],
            }
        )
    runs.sort(key=lambda r: r["time"])
    return runs


def render_table(runs: list[dict]) -> str:
    lines = [
        "| # | 运行时间 | 结果文件 | 用例集 | 模型 | 通过 | 失败用例 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(runs, start=1):
        failed = "、".join(r["failed_ids"]) if r["failed_ids"] else "无"
        lines.append(
            f"| {i} | {r['time']:%Y-%m-%d %H:%M:%S} | `{r['file']}` | {r['cases_file']} "
            f"| {r['model']} | {r['passed']}/{r['total']} | {failed} |"
        )
    return "\n".join(lines)


def render_section(runs: list[dict]) -> str:
    return "\n".join(
        [
            HEADING,
            "",
            MARKER,
            "",
            f"共 {len(runs)} 次运行，按时间排序。本节由 `python evals/summarize_results.py --append` "
            "从 `evals/results/*.json` 生成，上文的文字记录未做修改。",
            "结果文件当时没有记录所用的用例集与版本，「用例集」列为「未记录」；"
            "无法据此确认各轮运行时 cases.json 的具体内容。",
            "",
            render_table(runs),
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--append", metavar="FILE", help="把表格小节追加到该文件末尾")
    args = parser.parse_args()

    runs = load_runs()
    if not runs:
        print("evals/results/ 下没有结果文件", file=sys.stderr)
        sys.exit(1)

    if not args.append:
        print(render_table(runs))
        return

    with open(args.append, "rb") as f:
        existing = f.read()
    if MARKER.encode("utf-8") in existing:
        print(f"{args.append} 已包含本小节，未重复追加", file=sys.stderr)
        sys.exit(1)
    # 以二进制追加，保证原有字节不变；换行风格跟随原文件
    newline = b"\r\n" if b"\r\n" in existing else b"\n"
    section = render_section(runs).replace("\n", newline.decode())
    prefix = b"" if existing.endswith(newline) else newline
    with open(args.append, "ab") as f:
        f.write(prefix + newline + b"---" + newline + newline + section.encode("utf-8") + newline.decode().encode())
    print(f"已追加 {len(runs)} 条记录到 {args.append}")


if __name__ == "__main__":
    main()
