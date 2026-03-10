# -*- coding: utf-8 -*-
"""
Performance Analytics
Compares AI vs rule-based decisions, tracks trends, provides insights.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from utils import project_path


class PerformanceAnalytics:
    """
    Analyzes AI decision history and memory to provide performance insights.
    """

    def __init__(self, history_path: str = None,
                 memory_path: str = None):
        self.history_path = Path(history_path or project_path("data/decision_history.jsonl"))
        self.memory_path = Path(memory_path or project_path("data/ai_memory.json"))

    def get_dashboard_data(self) -> Dict:
        """Get complete dashboard data for GUI display."""
        history = self._load_history()
        memory = self._load_memory()

        return {
            'ai_vs_rules': self._ai_vs_rules_ratio(history),
            'boss_performance': self._boss_performance_summary(memory),
            'decision_breakdown': self._decision_type_breakdown(history),
            'hourly_activity': self._hourly_activity(history),
            'top_bosses': self._top_bosses(memory),
            'session_summary': self._session_summary(memory)
        }

    def _ai_vs_rules_ratio(self, history: List[Dict]) -> Dict:
        """Calculate AI vs rules decision ratio."""
        ai_count = sum(1 for h in history if h.get('source') == 'ai')
        rules_count = sum(1 for h in history if h.get('source') == 'rules')
        total = ai_count + rules_count

        return {
            'ai_decisions': ai_count,
            'rule_decisions': rules_count,
            'total': total,
            'ai_percentage': round(ai_count / max(1, total) * 100, 1)
        }

    def _boss_performance_summary(self, memory: Dict) -> Dict:
        """Summarize boss performance from memory."""
        bosses = memory.get('boss_performance', {})
        if not bosses:
            return {'total_hunts': 0, 'overall_success_rate': 0, 'avg_kill_time': 0}

        total_hunts = sum(b.get('total_hunts', 0) for b in bosses.values())
        total_success = sum(b.get('successful', 0) for b in bosses.values())
        kill_times = [b['avg_kill_time'] for b in bosses.values() if b.get('avg_kill_time', 0) > 0]

        return {
            'total_hunts': total_hunts,
            'overall_success_rate': round(total_success / max(1, total_hunts) * 100, 1),
            'avg_kill_time': round(sum(kill_times) / max(1, len(kill_times)), 1),
            'bosses_tracked': len(bosses)
        }

    def _decision_type_breakdown(self, history: List[Dict]) -> Dict:
        """Break down decisions by type."""
        breakdown = defaultdict(lambda: {'ai': 0, 'rules': 0})

        for entry in history:
            dtype = entry.get('type', 'unknown')
            source = entry.get('source', 'unknown')
            if source in ('ai', 'rules'):
                breakdown[dtype][source] += 1

        return dict(breakdown)

    def _hourly_activity(self, history: List[Dict]) -> Dict:
        """Track AI activity by hour of day."""
        hourly = defaultdict(int)

        for entry in history:
            try:
                ts = datetime.fromisoformat(entry.get('timestamp', ''))
                hourly[ts.hour] += 1
            except (ValueError, TypeError):
                pass

        return dict(hourly)

    def _top_bosses(self, memory: Dict) -> List[Dict]:
        """Get top performing bosses sorted by success rate."""
        bosses = memory.get('boss_performance', {})
        if not bosses:
            return []

        ranked = []
        for name, stats in bosses.items():
            hunts = stats.get('total_hunts', 0)
            if hunts == 0:
                continue
            success_rate = stats.get('successful', 0) / hunts
            ranked.append({
                'name': name,
                'hunts': hunts,
                'success_rate': round(success_rate * 100, 1),
                'avg_time': round(stats.get('avg_kill_time', 0), 1),
                'best_time': round(stats.get('best_time', 0), 1) if stats.get('best_time', float('inf')) != float('inf') else 0
            })

        return sorted(ranked, key=lambda x: x['success_rate'], reverse=True)

    def _session_summary(self, memory: Dict) -> Dict:
        """Get session summary from memory metadata."""
        meta = memory.get('metadata', {})
        return {
            'total_sessions': meta.get('total_sessions', 0),
            'total_decisions': meta.get('total_decisions', 0),
            'total_bosses_killed': meta.get('total_bosses_killed', 0)
        }

    def get_text_report(self) -> str:
        """Generate human-readable text report."""
        data = self.get_dashboard_data()
        lines = []
        lines.append("=" * 50)
        lines.append("  AI PERFORMANS RAPORU")
        lines.append("=" * 50)

        # AI vs Rules
        avr = data['ai_vs_rules']
        lines.append(f"\n  AI/Kural Orani: {avr['ai_percentage']}% AI")
        lines.append(f"  AI Karar: {avr['ai_decisions']} | Kural: {avr['rule_decisions']}")

        # Boss performance
        bp = data['boss_performance']
        lines.append(f"\n  Boss Avlari: {bp['total_hunts']} toplam")
        lines.append(f"  Basari: %{bp['overall_success_rate']}")
        lines.append(f"  Ort. Olum Suresi: {bp['avg_kill_time']}s")

        # Top bosses
        if data['top_bosses']:
            lines.append("\n  En Iyi Boss'lar:")
            for b in data['top_bosses'][:5]:
                lines.append(f"    {b['name']}: %{b['success_rate']} ({b['hunts']} av, ort {b['avg_time']}s)")

        # Session
        ss = data['session_summary']
        lines.append(f"\n  Oturum: {ss['total_sessions']} | Karar: {ss['total_decisions']} | Kill: {ss['total_bosses_killed']}")
        lines.append("=" * 50)

        return "\n".join(lines)

    def _load_history(self) -> List[Dict]:
        """Load decision history from JSONL file."""
        if not self.history_path.exists():
            return []

        entries = []
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception:
            pass
        return entries

    def _load_memory(self) -> Dict:
        """Load memory from JSON file."""
        if not self.memory_path.exists():
            return {}
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}


# Test code
if __name__ == "__main__":
    print("=== Performance Analytics Test ===\n")

    analytics = PerformanceAnalytics(
        history_path="../data/decision_history.jsonl",
        memory_path="../data/ai_memory.json"
    )

    report = analytics.get_text_report()
    print(report)

    print("\nDashboard data:")
    data = analytics.get_dashboard_data()
    for key, value in data.items():
        print(f"  {key}: {value}")
