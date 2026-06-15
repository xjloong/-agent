"""
批量作答脚本
======================
读取 question_public.csv 中的用户问题，逐条调用智能体生成回答，
输出符合 submission_example.csv 格式的提交文件。

用法：
    python code/batch_answer.py
    python code/batch_answer.py --input data/KownledgeBase/question_public.csv --output submission.csv

输出格式：
    id,ret
    1,回答文本
    2,回答文本
"""

import csv
import os
import sys
import time
import argparse
from typing import List, Tuple, Optional

# 确保能找到同目录下的其他模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer import generate_final_answer


def parse_questions_csv(file_path: str) -> List[Tuple[str, str]]:
    """
    解析 question_public.csv，返回 [(id, question_text), ...] 列表。

    CSV 特点：
    - utf-8 编码
    - question 字段可能跨多行（由双引号包裹）
    - id 可能不连续（如缺少 5, 27-32 等）
    """
    rows: List[Tuple[str, str]] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ['id', 'question']:
            print(f"⚠️  CSV 表头异常: {header}，期望 ['id', 'question']")

        for row in reader:
            # 跳过空行
            if not row or all(cell.strip() == '' for cell in row):
                continue
            # 正常情况下 row 有 2 列: [id, question]
            if len(row) >= 2:
                qid = row[0].strip()
                question = row[1].strip()
                rows.append((qid, question))
            else:
                print(f"⚠️  跳过异常行: {row}")

    print(f"✅ 成功读取 {len(rows)} 条问题（id 范围: {rows[0][0]} ~ {rows[-1][0]}）")
    return rows


def generate_answer(question: str, session_id: str = "batch") -> str:
    """
    调用智能体生成单条回答。

    使用 generate_final_answer 的同步路径，
    返回纯文本回答（图片引用已由 <PIC> 标记嵌入）。
    """
    try:
        answer_text, returned_images = generate_final_answer(
            question=question,
            images=[],
            session_id=session_id
        )
        return answer_text
    except Exception as e:
        print(f"❌ 问题生成失败: {e}")
        return "很抱歉，智能体暂时无法回答您的问题，请稍后再试。"


def write_submission_csv(
    results: List[Tuple[str, str]],
    output_path: str,
    append: bool = False
):
    """
    写入提交文件（CSV 格式：id,ret）。

    如果已有文件且 append=True，则合并（已存在的 id 保留原有结果，新增的追加）。
    """
    mode = 'a' if append else 'w'
    write_header = not (append and os.path.exists(output_path))

    with open(output_path, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['id', 'ret'])

        already_written = set()
        if append and os.path.exists(output_path):
            # 读已有文件，跳过 header
            with open(output_path, 'r', encoding='utf-8') as existing:
                reader = csv.reader(existing)
                next(reader, None)  # skip header
                for row in reader:
                    if row:
                        already_written.add(row[0])

        for qid, answer in results:
            if qid in already_written:
                continue  # 已存在则跳过（保护已有结果）
            # 如果回答含逗号或引号，csv.writer 自动加引号包裹
            writer.writerow([qid, answer])

    total = len(results)
    skipped = len([r for r in results if r[0] in already_written]) if append else 0
    written = total - skipped
    print(f"✅ 已写入 {written} 条结果到 {output_path}" + (f"（跳过 {skipped} 条已有记录）" if skipped else ""))


def batch_process(
    questions: List[Tuple[str, str]],
    output_path: str,
    resume: bool = False,
    delay: float = 0.0,
    max_retries: int = 2
):
    """
    批量处理所有问题，支持断点续传。

    Args:
        questions: [(id, question), ...]
        output_path: 输出 CSV 路径
        resume: 是否续传（跳过已处理的 id）
        delay: 每次调用间的延迟秒数（避免 API 限流）
        max_retries: 失败重试次数
    """
    if resume and os.path.exists(output_path):
        # 读取已处理过的 id
        processed_ids = set()
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    processed_ids.add(row[0].strip())
        remaining = [(qid, q) for qid, q in questions if qid not in processed_ids]
        print(f"📌 续传模式: {len(processed_ids)} 条已处理，剩余 {len(remaining)} 条")
    else:
        remaining = questions

    total = len(remaining)
    results: List[Tuple[str, str]] = []

    for idx, (qid, question) in enumerate(remaining, 1):
        print(f"\n[{idx}/{total}] id={qid} 处理中...")
        print(f"   问题: {question[:80]}{'...' if len(question) > 80 else ''}")

        answer = None
        for attempt in range(1 + max_retries):
            try:
                answer = generate_answer(question, session_id=f"batch_{qid}")
                break
            except Exception as e:
                print(f"   ⚠️  第 {attempt+1} 次尝试失败: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    answer = "很抱歉，智能体暂时无法回答您的问题，请稍后再试。"

        if answer:
            # 截断过长回答（留有余量）
            if len(answer) > 5000:
                answer = answer[:5000] + "……"
            print(f"   回答: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            results.append((qid, answer))

            # 实时写入，防止中断丢失
            write_submission_csv([(qid, answer)], output_path, append=True)

        if delay > 0 and idx < total:
            time.sleep(delay)

    print(f"\n🎉 批量处理完成！共成功生成 {len(results)} / {total} 条回答。")
    print(f"📄 提交文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="批量作答脚本 - 多模态客服智能体")
    parser.add_argument(
        "--input", "-i",
        default="data/KownledgeBase/question_public.csv",
        help="问题 CSV 文件路径（默认: data/KownledgeBase/question_public.csv）"
    )
    parser.add_argument(
        "--output", "-o",
        default="submission.csv",
        help="输出提交 CSV 文件路径（默认: submission.csv）"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="续传模式：跳过输出文件中已有 id，只处理未完成的"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.5,
        help="每次 API 调用间的延迟秒数（默认 0.5s）"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=0,
        help="限制处理条数（用于测试，0=全部）"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("🚀 多模态客服智能体 - 批量作答")
    print("=" * 50)

    # 1. 读取问题
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    questions = parse_questions_csv(args.input)

    # 2. 可选限制数量
    if args.limit > 0:
        questions = questions[:args.limit]
        print(f"🧪 测试模式: 仅处理前 {args.limit} 条")

    # 3. 批量处理
    batch_process(
        questions=questions,
        output_path=args.output,
        resume=args.resume,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
