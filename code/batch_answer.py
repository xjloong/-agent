"""
批量作答脚本
======================
读取 question_public.csv 中的用户问题，逐条调用智能体生成回答，
输出符合 submission_example.csv 格式的提交文件。

用法：
    python code/batch_answer.py
    python code/batch_answer.py --input data/KownledgeBase/question_public.csv --output submission.csv
    python code/batch_answer.py --resume                    # 手动指定续传

自动断点续传：
    - 运行中断后（Ctrl+C 或崩溃），重新运行会自动从上次中断处恢复
    - 通过输出文件（.csv）记录已写入的 id，进度文件（.progress）记录最后处理位置
    - 处理完全完成后自动清理进度文件

输出格式（id,ret[,图片列表]）：
    id,ret
    1,回答文本 <PIC>更多内容,["imgname1","imgname2"]
    2,回答文本

注意：回答中的换行符会在写入前自动去除，确保每条记录占一行。
有图片时附加第三列记录对应的图片名称列表。
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


def generate_answer(question: str, session_id: str = "batch") -> Tuple[str, List[str]]:
    """
    调用智能体生成单条回答。

    使用 generate_final_answer 的同步路径，
    返回 (回答文本, 图片名称列表) — 回答中的 <PIC> 标记与图片名称列表对应。
    """
    try:
        answer_text, returned_images = generate_final_answer(
            question=question,
            images=[],
            session_id=session_id
        )
        return answer_text, returned_images
    except Exception as e:
        print(f"❌ 问题生成失败: {e}")
        return "很抱歉，智能体暂时无法回答您的问题，请稍后再试。", []


def _build_ret_field(answer: str, images: List[str]) -> str:
    """
    构建 ret 字段的完整内容。

    格式遵循 格式正确回答.csv：
    - 有图片时：  "回答文本", ["img1","img2"]
    - 无图片时：  回答文本
    csv.writer 会在必要时自动对 ret 进行 CSV 引号转义。
    """
    import json
    ret = answer.replace('\n', '').replace('\r', '')
    if images:
        ret = f'"{ret}", {json.dumps(images, ensure_ascii=False)}'
    return ret


def write_submission_csv(
    results: List[Tuple[str, str, List[str]]],
    output_path: str,
    append: bool = False
):
    """
    写入提交文件（CSV 格式：id,ret — 共两列）。

    results 中每条为 (id, answer_text, image_names)。
    回答中的换行符会被去除，确保每条记录仅占一行。
    当有图片时，ret 字段格式为 "回答文本", ["img1","img2"]。
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

        for qid, answer, images in results:
            if qid in already_written:
                continue  # 已存在则跳过（保护已有结果）
            ret = _build_ret_field(answer, images)
            writer.writerow([qid, ret])

    total = len(results)
    skipped = len([r for r in results if r[0] in already_written]) if append else 0
    written = total - skipped
    print(f"✅ 已写入 {written} 条结果到 {output_path}" + (f"（跳过 {skipped} 条已有记录）" if skipped else ""))


CHECKPOINT_EXT = ".progress"


def get_checkpoint_path(output_path: str) -> str:
    """获取与输出文件关联的进度文件路径。"""
    return output_path + CHECKPOINT_EXT


def load_checkpoint(output_path: str) -> Optional[str]:
    """读取断点，返回最后一个已处理的 id（未找到返回 None）。"""
    cp_path = get_checkpoint_path(output_path)
    if os.path.exists(cp_path):
        with open(cp_path, 'r') as f:
            last_id = f.read().strip()
            return last_id if last_id else None
    return None


def save_checkpoint(output_path: str, qid: str):
    """保存断点：记录最后一个成功处理的 id。"""
    cp_path = get_checkpoint_path(output_path)
    os.makedirs(os.path.dirname(cp_path) or '.', exist_ok=True)
    with open(cp_path, 'w') as f:
        f.write(qid)


def get_processed_ids(output_path: str) -> set:
    """读取输出 CSV 中已处理过的 id 集合。"""
    if not os.path.exists(output_path):
        return set()
    processed = set()
    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0].strip():
                processed.add(row[0].strip())
    return processed


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
        resume: 是否续传（跳过已处理的 id）; 启动时自动检测已有输出/进度文件
        delay: 每次调用间的延迟秒数（避免 API 限流）
        max_retries: 失败重试次数
    """
    # Auto-detect: 检测到输出文件或进度文件时自动启用续传
    if not resume:
        if get_processed_ids(output_path) or load_checkpoint(output_path) is not None:
            print("📌 检测到已有输出/进度文件，自动启用续传模式")
            resume = True

    if resume:
        processed_ids = get_processed_ids(output_path)
        remaining = [(qid, q) for qid, q in questions if qid not in processed_ids]
        print(f"📌 续传模式: {len(processed_ids)} 条已处理，剩余 {len(remaining)} 条")
        if not remaining:
            print("✅ 所有问题已处理完毕！")
            return
    else:
        # 全新开始，清理旧进度文件
        cp_path = get_checkpoint_path(output_path)
        if os.path.exists(cp_path):
            os.remove(cp_path)
        remaining = questions

    total = len(remaining)
    results: List[Tuple[str, str, List[str]]] = []

    try:
        for idx, (qid, question) in enumerate(remaining, 1):
            print(f"\n[{idx}/{total}] id={qid} 处理中...")
            print(f"   问题: {question[:80]}{'...' if len(question) > 80 else ''}")

            answer = None
            images = []
            for attempt in range(1 + max_retries):
                try:
                    answer, images = generate_answer(question, session_id=f"batch_{qid}")
                    break
                except Exception as e:
                    print(f"   ⚠️  第 {attempt+1} 次尝试失败: {e}")
                    if attempt < max_retries:
                        time.sleep(2)
                    else:
                        answer = "很抱歉，智能体暂时无法回答您的问题，请稍后再试。"
                        images = []

            if answer:
                # 去除换行符，确保 CSV 每行对应一条记录
                answer = answer.replace('\n', '').replace('\r', '')
                # 截断过长回答（留有余量）
                if len(answer) > 5000:
                    answer = answer[:5000] + "……"
                if images:
                    print(f"   图片: {images}")
                print(f"   回答: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                results.append((qid, answer, images))

                # 实时写入 + 保存断点，防止中断丢失
                write_submission_csv([(qid, answer, images)], output_path, append=True)
                save_checkpoint(output_path, qid)

            if delay > 0 and idx < total:
                time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n⏸️  被用户中断！已处理 {len(results)}/{total} 条。")
        print(f"   重新运行将自动从断点处继续。")
        return

    # 处理完成，清理进度文件
    cp_path = get_checkpoint_path(output_path)
    if os.path.exists(cp_path):
        os.remove(cp_path)

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
