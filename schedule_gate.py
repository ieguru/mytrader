# schedule_gate.py
"""
'지금이 실행 시간인가'만 판정하는 게이트입니다.
워크플로가 매시간(:05) 이 스크립트를 돌리고, 통과할 때만 분석합니다.

사용
    python schedule_gate.py                  # 판정 후 결과를 GitHub Output에 기록
    python schedule_gate.py --dry-run-at "2026-11-19 20:05"
    python schedule_gate.py --force          # 무조건 실행(수동 실행용)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import schedule_config as sc

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "schedule.json"
STATE_PATH = BASE_DIR / "output" / "schedule_state.json"


def load_state(path=STATE_PATH):
    path = Path(path)

    if not path.is_file():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}


def save_state(state, path=STATE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_output(due, reason, slot, moment, config):
    """GitHub Actions 출력/요약에 기록합니다."""
    value = "true" if due else "false"

    output_file = os.environ.get("GITHUB_OUTPUT")

    if output_file:
        with open(output_file, "a", encoding="utf-8") as handle:
            handle.write(f"due={value}\n")
            handle.write(f"slot={slot or ''}\n")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")

    if summary_file:
        text = (
            f"### 스케줄 판정: {'실행' if due else '건너뜀'}\n"
            f"- 설정: {sc.describe(config)}\n"
            f"- 현재 시각(KST): {moment:%Y-%m-%d %H:%M}\n"
            f"- 판정: {reason}\n"
            f"- 슬롯: `{slot}`\n"
        )

        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write(text)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="스케줄 게이트")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--state", default=str(STATE_PATH))
    parser.add_argument(
        "--dry-run-at",
        default=None,
        help='판정 기준 시각 재정의 (예: "2026-11-19 20:05")',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="설정과 무관하게 실행합니다.",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="상태 파일을 읽거나 쓰지 않습니다.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        config = sc.load_config(args.config)

    except sc.ScheduleConfigError as error:
        print(f"[스케줄] 설정 오류: {error}")
        print("[스케줄] 기본값(매일 20:01)으로 판정합니다.")
        config = dict(sc.DEFAULT_CONFIG)

    moment = sc.as_kst(
        datetime.fromisoformat(args.dry_run_at)
        if args.dry_run_at
        else None
    )

    if args.force:
        due, reason, slot = True, "수동 강제 실행", "force"
        print(f"[스케줄] 강제 실행 — {sc.describe(config)}")

    else:
        state = load_state(args.state) if not args.no_state else {}
        due, reason, slot = sc.is_due_with_state(
            config,
            moment,
            state.get("last_slot"),
        )

        print(f"[스케줄] 설정: {sc.describe(config)}")
        print(f"[스케줄] 현재(KST): {moment:%Y-%m-%d %H:%M} "
              f"({sc.WEEKDAY_NAMES[moment.weekday()]})")
        print(f"[스케줄] 판정: {'실행' if due else '건너뜀'} — {reason}")

        upcoming = sc.next_runs(config, moment, count=3)

        if upcoming:
            print(
                "[스케줄] 다음 예정: "
                + ", ".join(
                    f"{item:%m-%d %H:%M}" for item in upcoming
                )
            )

        if due and not args.no_state:
            # 실행 직전에 슬롯을 기록해 같은 시간대 중복 실행을 막습니다.
            save_state({"last_slot": slot, "saved_at": moment.isoformat()},
                       args.state)

    write_output(due, reason, slot, moment, config)

    return 0 if due else 1


if __name__ == "__main__":
    raise SystemExit(main())
