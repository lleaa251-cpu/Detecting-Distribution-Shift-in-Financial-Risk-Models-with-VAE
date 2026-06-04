#!/usr/bin/env python3
"""
VAE 監控潛在空間分佈偵測資料偏移
金融風控模型資料漂移即時監控系統

使用方式：
  python run_all.py            # 執行全部 5 個 Phase
  python run_all.py --phase 3  # 只執行 Phase 3（需先完成前置 Phase）
  python run_all.py --phase 1 2 3  # 執行指定多個 Phase

環境需求：
  pip install numpy pandas matplotlib scikit-learn scipy
"""

import sys
import os
import time
import argparse

# 加入當前目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║   VAE Distribution Drift Monitor — Financial Risk Control       ║
║   基於變分自編碼器的金融風控模型資料漂移即時監控系統              ║
╚══════════════════════════════════════════════════════════════════╝
"""

PHASE_INFO = {
    1: ("資料準備 + 分佈偏移注入",    "phase1_data_prep"),
    2: ("訓練 Transformer Classifier", "phase2_classifier"),
    3: ("訓練 VAE 監控器",            "phase3_vae"),
    4: ("偏移偵測機制",               "phase4_detection"),
    5: ("t-SNE 視覺化 + 再訓練觸發",  "phase5_visualization"),
}


def run_phase(phase_num):
    name, module = PHASE_INFO[phase_num]
    print(f"\n{'─'*60}")
    print(f"▶  Phase {phase_num}: {name}")
    print(f"{'─'*60}")
    start = time.time()
    __import__(module)
    elapsed = time.time() - start
    print(f"✓  Phase {phase_num} 完成（耗時 {elapsed:.1f}s）")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description='VAE Distribution Drift Monitoring System'
    )
    parser.add_argument(
        '--phase', type=int, nargs='+',
        choices=[1, 2, 3, 4, 5],
        help='執行指定 Phase（預設執行全部）'
    )
    args = parser.parse_args()

    print(BANNER)
    phases_to_run = args.phase if args.phase else [1, 2, 3, 4, 5]
    print(f"將執行 Phase: {phases_to_run}")

    os.makedirs('output', exist_ok=True)
    total_start = time.time()
    timings = {}

    for phase in phases_to_run:
        try:
            elapsed = run_phase(phase)
            timings[phase] = elapsed
        except FileNotFoundError as e:
            print(f"\n❌ Phase {phase} 失敗：{e}")
            print(f"   請先執行前置 Phase（例如：python run_all.py --phase {phase-1}）")
            sys.exit(1)
        except Exception as e:
            import traceback
            print(f"\n❌ Phase {phase} 發生錯誤：{e}")
            traceback.print_exc()
            sys.exit(1)

    total = time.time() - total_start

    print(f"\n{'═'*60}")
    print("執行摘要")
    print(f"{'═'*60}")
    for phase, t in timings.items():
        name, _ = PHASE_INFO[phase]
        print(f"  Phase {phase}: {name:<30} {t:.1f}s")
    print(f"{'─'*60}")
    print(f"  總計耗時: {total:.1f}s")

    # 列出輸出檔案
    print(f"\n輸出檔案（output/）：")
    if os.path.exists('output'):
        for f in sorted(os.listdir('output')):
            fp = os.path.join('output', f)
            size = os.path.getsize(fp)
            print(f"  {f:<45} {size/1024:.1f} KB")

    print(f"\n{'═'*60}")
    print("✓  全部完成！請查看 output/ 資料夾中的圖表與模型")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
