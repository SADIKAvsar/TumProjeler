# -*- coding: utf-8 -*-
"""
Memory Manager
JSON-based experience database for AI learning.
Tracks boss performance, strategic decisions, and error recovery outcomes.
"""

import json
import os
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import shutil
from utils import project_path


class MemoryManager:
    """
    Manages AI experience memory with thread-safe operations.
    Stores boss performance stats, decision outcomes, and learning data.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize memory manager.

        Args:
            db_path: Path to JSON memory file
        """
        if db_path:
            resolved_db_path = db_path
        else:
            preferred = Path(r"E:\LoABot_Training_Data\runtime_data\ai_memory.json")
            resolved_db_path = str(preferred) if preferred.parent.exists() else project_path("data/ai_memory.json")

        self.db_path = Path(resolved_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.memory = self.load_memory()
        # Re-entrant lock: update_* -> auto_save_if_needed -> save_memory
        # zincirinde ayni thread lock'u tekrar alabildigi icin deadlock olusmaz.
        self.lock = threading.RLock()

        # Auto-save timer
        self.last_save_time = time.time()
        self.save_interval = 60  # 1 dakika: NVMe SSD icin guvenli; veri kaybi riskini 5 dk'dan 60s'e dusuruyor

        # Backup configuration
        self.backup_interval = 86400  # 24 hours
        self.last_backup_time = time.time()
        self.max_backups = 7

    def load_memory(self) -> Dict:
        """
        Load memory from JSON file.

        Returns:
            Memory dict or empty structure if file doesn't exist
        """
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Memory] Failed to load {self.db_path}: {e}")
                # Try to load backup
                backup = self._find_latest_backup()
                if backup:
                    print(f"[Memory] Loading backup: {backup}")
                    with open(backup, 'r', encoding='utf-8') as f:
                        return json.load(f)

        return self.create_empty_memory()

    def create_empty_memory(self) -> Dict:
        """Create empty memory structure."""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "boss_performance": {},
            "strategic_decisions": {},
            "error_recovery": {},
            "event_priorities": {},
            "daily_stats": {},
            "metadata": {
                "total_sessions": 0,
                "total_decisions": 0,
                "total_bosses_killed": 0
            }
        }

    def save_memory(self):
        """
        Save memory to JSON file (thread-safe).
        """
        with self.lock:
            try:
                self.memory["last_updated"] = datetime.now().isoformat()

                # Write to temporary file first
                temp_path = self.db_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.memory, f, indent=2, ensure_ascii=False)

                # Atomic rename
                temp_path.replace(self.db_path)

                self.last_save_time = time.time()

                # Check if backup needed
                if time.time() - self.last_backup_time > self.backup_interval:
                    self._create_backup()

            except Exception as e:
                print(f"[Memory] Failed to save: {e}")

    def auto_save_if_needed(self):
        """Save memory if enough time has passed."""
        if time.time() - self.last_save_time > self.save_interval:
            self.save_memory()

    def _create_backup(self):
        """Create timestamped backup of memory file."""
        if not self.db_path.exists():
            return

        backup_dir = self.db_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"ai_memory_{timestamp}.json"

        shutil.copy2(self.db_path, backup_path)
        self.last_backup_time = time.time()

        # Clean old backups
        self._cleanup_old_backups(backup_dir)

    def _cleanup_old_backups(self, backup_dir: Path):
        """Keep only the most recent backups."""
        backups = sorted(backup_dir.glob("ai_memory_*.json"), reverse=True)
        for old_backup in backups[self.max_backups:]:
            old_backup.unlink()

    def _find_latest_backup(self) -> Optional[Path]:
        """Find most recent backup file."""
        backup_dir = self.db_path.parent / "backups"
        if not backup_dir.exists():
            return None

        backups = sorted(backup_dir.glob("ai_memory_*.json"), reverse=True)
        return backups[0] if backups else None

    # ===== Boss Performance Tracking =====

    def update_boss_performance(
        self,
        boss_name: str,
        kill_time: float,
        success: bool,
        loot_quality: Optional[float] = None
    ):
        """
        Update boss hunt performance statistics.

        Args:
            boss_name: Boss identifier
            kill_time: Time taken to kill boss (seconds)
            success: Whether hunt was successful
            loot_quality: Optional loot quality score (0-10)
        """
        with self.lock:
            if boss_name not in self.memory['boss_performance']:
                self.memory['boss_performance'][boss_name] = {
                    'total_hunts': 0,
                    'successful': 0,
                    'death_count': 0,
                    'avg_kill_time': 0.0,
                    'best_time': float('inf'),
                    'worst_time': 0.0,
                    'total_time': 0.0,
                    'loot_quality_avg': 0.0,
                    'loot_quality_samples': 0,
                    'last_hunt': None,
                    'preferred_time_of_day': []
                }

            stats = self.memory['boss_performance'][boss_name]
            stats['total_hunts'] += 1
            stats['last_hunt'] = datetime.now().isoformat()

            if success:
                stats['successful'] += 1
                stats['total_time'] += kill_time
                stats['avg_kill_time'] = stats['total_time'] / stats['successful']
                stats['best_time'] = min(stats['best_time'], kill_time)
                stats['worst_time'] = max(stats['worst_time'], kill_time)

                if loot_quality is not None:
                    prev_avg = stats['loot_quality_avg']
                    n = stats['loot_quality_samples']
                    stats['loot_quality_avg'] = (prev_avg * n + loot_quality) / (n + 1)
                    stats['loot_quality_samples'] += 1

                # Track time of day preference
                hour = datetime.now().hour
                stats['preferred_time_of_day'].append(hour)
                if len(stats['preferred_time_of_day']) > 100:
                    stats['preferred_time_of_day'] = stats['preferred_time_of_day'][-100:]
            else:
                stats['death_count'] += 1

            # Update metadata
            if success:
                self.memory['metadata']['total_bosses_killed'] += 1

            self.auto_save_if_needed()

    def get_boss_stats(self, boss_name: Optional[str] = None) -> Dict:
        """
        Get boss performance statistics.

        Args:
            boss_name: Specific boss name, or None for all bosses

        Returns:
            Boss stats dict
        """
        with self.lock:
            if boss_name:
                return self.memory['boss_performance'].get(boss_name, {})
            return self.memory['boss_performance'].copy()

    def get_boss_success_rate(self, boss_name: str) -> float:
        """Get success rate for specific boss (0.0 - 1.0)."""
        stats = self.get_boss_stats(boss_name)
        if not stats or stats['total_hunts'] == 0:
            return 0.5  # Default: assume 50% if no data
        return stats['successful'] / stats['total_hunts']

    # ===== Strategic Decision Tracking =====

    def record_strategic_decision(
        self,
        decision_type: str,
        outcome: bool,
        time_saved: float = 0.0,
        context: Optional[Dict] = None
    ):
        """
        Record outcome of a strategic decision.

        Args:
            decision_type: Type of decision (e.g., "wait_same_map")
            outcome: Whether decision was beneficial
            time_saved: Time saved/lost (positive = saved, negative = lost)
            context: Optional context dict
        """
        with self.lock:
            if decision_type not in self.memory['strategic_decisions']:
                self.memory['strategic_decisions'][decision_type] = {
                    'times_chosen': 0,
                    'successful_outcome': 0,
                    'total_time_impact': 0.0,
                    'contexts': []
                }

            stats = self.memory['strategic_decisions'][decision_type]
            stats['times_chosen'] += 1

            if outcome:
                stats['successful_outcome'] += 1

            stats['total_time_impact'] += time_saved

            if context:
                stats['contexts'].append({
                    'timestamp': datetime.now().isoformat(),
                    'outcome': outcome,
                    'time_impact': time_saved,
                    **context
                })
                # Keep only last 50 contexts
                if len(stats['contexts']) > 50:
                    stats['contexts'] = stats['contexts'][-50:]

            self.auto_save_if_needed()

    def get_strategic_wait_stats(
        self,
        current_boss: Optional[str] = None,
        next_boss: Optional[str] = None
    ) -> Dict:
        """Get strategic wait decision statistics."""
        with self.lock:
            key = "wait_same_map"
            if key not in self.memory['strategic_decisions']:
                return {
                    'times_chosen': 0,
                    'success_rate': 0.5,
                    'avg_time_saved': 0.0
                }

            stats = self.memory['strategic_decisions'][key]
            success_rate = stats['successful_outcome'] / max(1, stats['times_chosen'])
            avg_time_saved = stats['total_time_impact'] / max(1, stats['times_chosen'])

            return {
                'times_chosen': stats['times_chosen'],
                'success_rate': success_rate,
                'avg_time_saved': avg_time_saved
            }

    # ===== Error Recovery Tracking =====

    def record_freeze_diagnosis(
        self,
        diagnosis: Dict,
        duration: float,
        was_correct: Optional[bool] = None
    ):
        """
        Record freeze diagnosis and outcome.

        Args:
            diagnosis: AI diagnosis dict
            duration: Freeze duration (seconds)
            was_correct: Whether diagnosis was correct (if known)
        """
        with self.lock:
            if 'freeze_diagnoses' not in self.memory['error_recovery']:
                self.memory['error_recovery']['freeze_diagnoses'] = []

            record = {
                'timestamp': datetime.now().isoformat(),
                'duration': duration,
                'diagnosis': diagnosis,
                'was_correct': was_correct
            }

            self.memory['error_recovery']['freeze_diagnoses'].append(record)

            # Keep only last 100 diagnoses
            if len(self.memory['error_recovery']['freeze_diagnoses']) > 100:
                self.memory['error_recovery']['freeze_diagnoses'] = \
                    self.memory['error_recovery']['freeze_diagnoses'][-100:]

            self.auto_save_if_needed()

    # ===== Session Management =====

    def start_session(self):
        """Record new bot session."""
        with self.lock:
            self.memory['metadata']['total_sessions'] += 1
            self.memory['metadata']['last_session_start'] = datetime.now().isoformat()
            self.save_memory()

    def record_decision(self, decision_type: str):
        """Increment decision counter."""
        with self.lock:
            self.memory['metadata']['total_decisions'] += 1

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for display."""
        with self.lock:
            return {
                'total_sessions': self.memory['metadata'].get('total_sessions', 0),
                'total_decisions': self.memory['metadata'].get('total_decisions', 0),
                'total_bosses_killed': self.memory['metadata'].get('total_bosses_killed', 0),
                'bosses_tracked': len(self.memory['boss_performance']),
                'last_updated': self.memory.get('last_updated', 'Never')
            }


# Test code
if __name__ == "__main__":
    print("=== Memory Manager Test ===\n")

    # Initialize
    manager = MemoryManager("../data/ai_memory_test.json")
    print(f"✅ Memory manager initialized\n")

    # Test boss performance tracking
    print("Testing boss performance tracking...")
    manager.update_boss_performance("boss_800", kill_time=35.2, success=True, loot_quality=8.5)
    manager.update_boss_performance("boss_800", kill_time=42.1, success=True, loot_quality=7.2)
    manager.update_boss_performance("boss_800", kill_time=0.0, success=False)

    stats = manager.get_boss_stats("boss_800")
    print(f"  Boss 800 stats: {json.dumps(stats, indent=2)}")
    print(f"  Success rate: {manager.get_boss_success_rate('boss_800'):.1%}\n")

    # Test strategic decisions
    print("Testing strategic decision tracking...")
    manager.record_strategic_decision("wait_same_map", outcome=True, time_saved=45.0)
    manager.record_strategic_decision("wait_same_map", outcome=True, time_saved=38.5)
    manager.record_strategic_decision("wait_same_map", outcome=False, time_saved=-15.0)

    wait_stats = manager.get_strategic_wait_stats()
    print(f"  Wait stats: {json.dumps(wait_stats, indent=2)}\n")

    # Save
    manager.save_memory()
    print(f"✅ Memory saved to: {manager.db_path}")

    # Summary
    print("\n" + "="*50)
    summary = manager.get_summary_stats()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("="*50)
