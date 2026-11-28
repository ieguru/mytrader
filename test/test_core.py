"""핵심 계산 규칙 회귀 테스트.

표준 라이브러리 unittest만 사용합니다(별도 설치 불필요).

    python -m unittest discover -s tests -t .

검증 대상
- 스케줄 판정/슬롯 중복/UTC cron 변환
- PER: '기준일까지 종료된 분기'만 사용하고, 값이 비면 대체하지 않음
- 신호 통계: 방향별 기준선과 요약(승률·초과) 계산
- 스냅샷 직렬화 왕복
"""

from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd

import analysis_engine as ae
import per_status
import result_store
import schedule_config as sc


class ScheduleConfigTest(unittest.TestCase):
    def test_daily_due_only_at_selected_hour(self):
        config = dict(sc.DEFAULT_CONFIG, period="daily", hour=20, minute=1)

        due, _ = sc.is_due(config, datetime(2026, 11, 19, 20, 5, tzinfo=sc.KST))
        not_due, _ = sc.is_due(config, datetime(2026, 11, 19, 19, 5, tzinfo=sc.KST))

        self.assertTrue(due)
        self.assertFalse(not_due)

    def test_slot_dedup_blocks_second_run(self):
        config = dict(sc.DEFAULT_CONFIG, period="daily", hour=20)
        moment = datetime(2026, 11, 19, 20, 5, tzinfo=sc.KST)
        slot = sc.slot_key(config, moment)

        due, reason, _ = sc.is_due_with_state(config, moment, slot)

        self.assertFalse(due)
        self.assertIn("이미 이 슬롯", reason)

    def test_weekly_next_runs_land_on_weekday(self):
        config = dict(sc.DEFAULT_CONFIG, period="weekly", weekday=2, hour=20)
        runs = sc.next_runs(
            config,
            datetime(2026, 11, 19, 10, 0, tzinfo=sc.KST),
            count=2,
        )

        self.assertEqual(len(runs), 2)
        self.assertTrue(all(run.weekday() == 2 for run in runs))

    def test_cron_expression_is_utc(self):
        config = dict(sc.DEFAULT_CONFIG, period="daily", hour=20, minute=1)

        self.assertEqual(sc.cron_expression(config), "1 11 * * *")

    def test_invalid_period_rejected(self):
        with self.assertRaises(sc.ScheduleConfigError):
            sc.normalize_config({"period": "yearly"})


class PerStatusTest(unittest.TestCase):
    @staticmethod
    def _financial(rows):
        return pd.DataFrame({
            "분기": [row[0] for row in rows],
            "_분기종료일": pd.to_datetime([row[1] for row in rows]),
            "상장주식수": [row[2] for row in rows],
            "4Q_income": [row[3] for row in rows],
        })

    def test_uses_last_completed_quarter(self):
        financial = self._financial([
            ("2026.2Q", "2026-06-30", 1_000_000_000.0, 1000.0),
            ("2026.3Q", "2026-09-30", 1_000_000_000.0, 2000.0),
        ])

        status = per_status.build_per_status(
            financial,
            pd.Timestamp("2026-10-15"),
            10000.0,
        )

        self.assertEqual(status["PER 분기"], "2026.3Q")
        # 시총 10,000원 × 10억주 = 10조원 ÷ 순이익 2,000억원 = 50
        self.assertAlmostEqual(status["PER"], 50.0)

    def test_future_quarter_is_ignored(self):
        financial = self._financial([
            ("2026.3Q", "2026-09-30", 1_000_000_000.0, 2000.0),
            ("2026.4Q", "2026-12-31", 1_000_000_000.0, 4000.0),
        ])

        status = per_status.build_per_status(
            financial,
            pd.Timestamp("2026-11-19"),
            10000.0,
        )

        self.assertEqual(status["PER 분기"], "2026.3Q")
        self.assertAlmostEqual(status["PER"], 50.0)

    def test_missing_latest_income_does_not_fall_back(self):
        financial = self._financial([
            ("2026.2Q", "2026-06-30", 1_000_000_000.0, 1000.0),
            ("2026.3Q", "2026-09-30", 1_000_000_000.0, None),
        ])

        status = per_status.build_per_status(
            financial,
            pd.Timestamp("2026-10-15"),
            10000.0,
        )

        self.assertIsNone(status["PER"])

    def test_to_finite_number_rejects_noise(self):
        self.assertEqual(per_status.to_finite_number("1,234.5"), 1234.5)
        self.assertIsNone(per_status.to_finite_number("nan"))
        self.assertIsNone(per_status.to_finite_number(""))
        self.assertIsNone(per_status.to_finite_number(True))


class SignalStatsTest(unittest.TestCase):
    def test_baseline_counts_both_directions(self):
        # 30일 내내 같은 가격이면 비용 때문에 어느 방향도 승리가 아님
        prices = [100.0] * 30
        baselines = ae._directional_baseline(prices, prices, [5])

        self.assertEqual(baselines[1][5], 0.0)
        self.assertEqual(baselines[-1][5], 0.0)

    def test_horizon_longer_than_data_gives_none(self):
        prices = [100.0] * 3
        baselines = ae._directional_baseline(prices, prices, [5])

        self.assertIsNone(baselines[1][5])

    def test_summarize_records(self):
        records = [
            {"수익률5": 10.0},
            {"수익률5": -5.0},
            {"수익률5": None},
        ]

        summary = ae._summarize_records(records, [5], {5: 40.0})[0]

        self.assertEqual(summary["표본"], 2)
        self.assertEqual(summary["승률"], 50.0)
        self.assertEqual(summary["평균"], 2.5)
        self.assertEqual(summary["최선"], 10.0)
        self.assertEqual(summary["최악"], -5.0)
        self.assertEqual(summary["초과"], 10.0)

    def test_summarize_empty_records(self):
        summary = ae._summarize_records([], [5], {5: None})[0]

        self.assertEqual(summary["표본"], 0)
        self.assertIsNone(summary["승률"])
        self.assertIsNone(summary["초과"])


class ResultStoreTest(unittest.TestCase):
    def test_frame_roundtrip_keeps_dtypes(self):
        frame = pd.DataFrame({"승률": [60.0], "표본": [3]})

        back = result_store.json_to_frame(result_store.frame_to_json(frame))

        self.assertEqual(back["표본"].iat[0], 3)
        self.assertAlmostEqual(back["승률"].iat[0], 60.0)

    def test_dumps_loads_roundtrip(self):
        payload = {
            "version": result_store.SNAPSHOT_VERSION,
            "saved_at": "2026-11-19T20:05:00+09:00",
        }

        self.assertEqual(
            result_store.loads(result_store.dumps(payload)),
            payload,
        )

    def test_snapshot_requires_results(self):
        with self.assertRaises(result_store.SnapshotError):
            result_store.build_snapshot([], [], None)


if __name__ == "__main__":
    unittest.main()
