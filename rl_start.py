# -*- coding: utf-8 -*-
"""
rl_start.py : Visual Studio 2022에서 Ctrl+F5로 단계별 실행하는 시작 파일

사용법
  1) 아래 STEP 값만 바꾸고 저장한 뒤 Ctrl+F5
     예) STEP = 12           -> 12번 단계만 실행
         STEP = [10, 11, 12] -> 10, 11, 12번을 차례로 실행 (중간에 실패하면 멈춤)
  2) KOSPI200으로 실행하려면 UNIVERSE = "kospi200"
     (지원하는 단계에 "--universe kospi200"이 자동으로 앞에 붙음)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# ===================== 여기만 바꾸세요 =====================
STEP = 23
UNIVERSE = ""          # "" = 기존 41종목(stocks.txt), "kospi200" = KOSPI200
PAUSE_AT_END = True    # 끝난 뒤 창이 바로 닫히지 않도록 Enter 대기
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# 번호: (실행할 모듈, 인자 목록, 설명)
COMMANDS = {
    # ---- 데이터 준비 ----
    0:  ("rl.download", ["--universe", "stocks", "--start", "2015-01-01"],
         "가격·수급 데이터 다운로드 (이미 받은 종목은 건너뜀)"),
    1:  ("rl.run", ["--prepare-only"],
         "지표·point-in-time 신호 계산 (processed/ 생성)"),
    2:  ("pytest", ["-q", "tests/test_lookahead.py"],
         "미래 정보 자동 검사 (1 passed 확인)"),

    # ---- 매일 결정 PPO (기존 실험) ----
    3:  ("rl.run", ["--debug"],
         "초소형 디버그"),
    4:  ("rl.run", ["--folds", "F2024", "--seeds", "1", "--timesteps", "60000"],
         "빠른 점검"),
    5:  ("rl.run", ["--seeds", "5", "--reward", "dd", "--no-tb"],
         "본 실험 (--no-tb 미적용 시 그 항목 삭제)"),

    # ---- 주간 매수 신호 필터 (run_filter) ----
    6:  ("rl.run_filter", ["--check"], "필터: 미래 정보 점검"),
    7:  ("rl.run_filter", ["--debug"], "필터: 디버그"),
    8:  ("rl.run_filter", [], "필터: 본 실험"),
    9:  ("rl.run_filter", ["--test-years", "2020", "2021", "2022", "2023",
                           "2024", "2025", "2026", "--tag", "_7y"],
         "필터: 7개 연도 시험"),

    # ---- 매수·매도 순서 유지 필터 (run_filter_seq) ----
    10: ("rl.run_filter_seq", ["--check"], "순서 필터: 미래 정보 점검"),
    11: ("rl.run_filter_seq", ["--debug"], "순서 필터: 디버그"),
    12: ("rl.run_filter_seq", [], "순서 필터: 본 실험"),
    13: ("rl.run_filter_seq", ["--test-years", "2020", "2021", "2022", "2023",
                               "2024", "2025", "2026", "--tag", "_7y"],
         "순서 필터: 7개 연도 시험"),

    # ---- 진단 / 보상 비교 ----
    14: ("rl.diag_signal", [], "매수 신호 성패 예측력 진단 (AUC)"),
    15: ("rl.run_filter_seq", ["--reward", "excess", "--tag", "_excess"],
         "보상: 시장 초과수익"),
    16: ("rl.run_filter_seq", ["--reward", "excess_risk", "--tag", "_excess_risk"],
         "보상: 초과수익 + 변동성 조정"),
    17: ("rl.diag_features", [], "후보 피처 IC 진단"),

    # ---- 전략 정리 (전략 1~3 + 주간 신호 비교) ----
    18: ("rl.run_filter_seq", ["--signals-only", "--start", "2016-01-01"],
         "신호 전략 비교표 (단순보유/이동평균/이동평균(RSI)/주간)"),

    # ---- 추가 피처 ----
    19: ("rl.download_extra", [], "PER·PBR·시총·공매도·외국인지분율 다운로드"),
    20: ("rl.run_filter_seq", ["--diag"], "피처별 연도별 IC 진단"),
    21: ("rl.run_filter_seq", ["--tag", "_extra"], "추가 피처 포함 본 실험"),
    22: ("rl.run_filter_seq", ["--no-extra", "--tag", "_base"], "기본 피처만 (비교용)"),
    23: ("rl.run_filter_seq", ["--reward", "risk", "--tag", "_risk"],
         "보상: 변동성 조정만"),
}

# --universe 를 받지 않는 단계
NO_UNIVERSE = {2}


def build_command(step: int) -> list:
    module, args, _ = COMMANDS[step]
    args = list(args)
    if UNIVERSE and step not in NO_UNIVERSE:
        if "--universe" in args:                      # 0번처럼 이미 있으면 값만 교체
            args[args.index("--universe") + 1] = UNIVERSE
        else:
            args = ["--universe", UNIVERSE] + args
        if UNIVERSE != "stocks" and "--tag" not in args and step not in (0, 1, 19):
            args += ["--tag", f"_{UNIVERSE}"]         # 결과 폴더 덮어쓰기 방지
    return [sys.executable, "-m", module] + args


def run_step(step: int) -> int:
    if step not in COMMANDS:
        print(f"[오류] STEP {step} 은(는) 없습니다. 가능한 번호: {sorted(COMMANDS)}")
        return 2
    cmd = build_command(step)
    print("=" * 70)
    print(f"STEP {step}: {COMMANDS[step][2]}")
    print("실행:", " ".join(cmd[1:]))
    print("=" * 70, flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"     # 한글 출력 깨짐 방지
    env["PYTHONUTF8"] = "1"

    t0 = time.time()
    code = subprocess.call(cmd, cwd=str(BASE_DIR), env=env)
    mins = (time.time() - t0) / 60
    status = "완료" if code == 0 else f"실패 (종료 코드 {code})"
    print(f"\n>>> STEP {step} {status} — {mins:.1f}분 소요\n", flush=True)
    return code


def main() -> int:
    steps = STEP if isinstance(STEP, (list, tuple)) else [STEP]
    print(f"프로젝트 폴더: {BASE_DIR}")
    print(f"파이썬: {sys.executable}")
    print(f"유니버스: {UNIVERSE or 'stocks (기본 41종목)'}\n")

    code = 0
    for s in steps:
        code = run_step(int(s))
        if code != 0:
            print("오류가 발생해 이후 단계는 실행하지 않습니다.")
            print("콘솔 마지막 10~20줄을 복사해 보내주세요.")
            break

    if PAUSE_AT_END:
        try:
            input("Enter 키를 누르면 창이 닫힙니다...")
        except EOFError:
            pass
    return code


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
