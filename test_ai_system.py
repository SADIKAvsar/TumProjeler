# -*- coding: utf-8 -*-
"""
AI System Test Script
Verifies that all AI components are working correctly.
"""

import sys
import os

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LoA BoT v5.9"))

import time
import numpy as np
from gemini_client import GeminiClient
from memory_manager import MemoryManager
from ai_engine import AIEngine, GeminiRateLimiter, AIDecisionCache
import yaml


def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def test_config():
    """Test 1: Configuration Loading"""
    print_header("TEST 1: Configuration Loading")

    try:
        with open("config/ai_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print("✅ Config file loaded successfully")

        api_key = config['ai_settings']['gemini']['api_key']
        if api_key == "YOUR_API_KEY_HERE":
            print("❌ API key not set! Please update config/ai_config.yaml")
            return False

        print(f"✅ API key found: {api_key[:10]}...")
        print(f"   Model: {config['ai_settings']['gemini']['model_name']}")
        print(f"   Features enabled: {config['ai_settings']['features']}")

        return True

    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_gemini_client():
    """Test 2: Gemini Client"""
    print_header("TEST 2: Gemini Client Connection")

    try:
        with open("config/ai_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        api_key = config['ai_settings']['gemini']['api_key']
        model_name = config['ai_settings']['gemini']['model_name']

        client = GeminiClient(api_key, model_name)
        print(f"✅ Gemini client initialized")

        # Test connection
        print("   Testing API connection...")
        success, message = client.test_connection()

        if success:
            print(f"✅ {message}")

            stats = client.get_statistics()
            print(f"\n   Statistics:")
            print(f"   - Calls made: {stats['total_calls']}")
            print(f"   - Tokens used: {stats['tokens_input']} in, {stats['tokens_output']} out")
            print(f"   - Cost: ${stats['cost_usd']}")

            return True
        else:
            print(f"❌ {message}")
            return False

    except Exception as e:
        print(f"❌ Gemini client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_manager():
    """Test 3: Memory Manager"""
    print_header("TEST 3: Memory Manager")

    try:
        manager = MemoryManager("data/ai_memory_test.json")
        print("✅ Memory manager initialized")

        # Test boss performance tracking
        print("   Testing boss performance tracking...")
        manager.update_boss_performance("test_boss", kill_time=35.5, success=True, loot_quality=8.0)
        manager.update_boss_performance("test_boss", kill_time=42.0, success=True, loot_quality=7.5)
        manager.update_boss_performance("test_boss", kill_time=0.0, success=False)

        stats = manager.get_boss_stats("test_boss")
        print(f"   - Total hunts: {stats['total_hunts']}")
        print(f"   - Success rate: {stats['successful']}/{stats['total_hunts']}")
        print(f"   - Avg kill time: {stats['avg_kill_time']:.1f}s")

        # Test strategic decisions
        print("\n   Testing strategic decision tracking...")
        manager.record_strategic_decision("test_decision", outcome=True, time_saved=45.0)
        manager.record_strategic_decision("test_decision", outcome=False, time_saved=-10.0)

        # Save
        manager.save_memory()
        print(f"✅ Memory saved to: {manager.db_path}")

        summary = manager.get_summary_stats()
        print(f"\n   Summary:")
        for key, value in summary.items():
            print(f"   - {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rate_limiter():
    """Test 4: Rate Limiter"""
    print_header("TEST 4: Rate Limiter")

    try:
        with open("config/ai_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        limiter = GeminiRateLimiter(config['ai_settings']['rate_limits'])
        print("✅ Rate limiter initialized")

        # Check if can make call
        can_call, reason = limiter.can_make_call()
        if can_call:
            print(f"✅ Can make API call: {reason}")
        else:
            print(f"⚠️ Cannot make call: {reason}")

        # Simulate some calls
        print("\n   Simulating 3 API calls...")
        for i in range(3):
            limiter.record_call(tokens_in=1500, tokens_out=200, images=1)

        stats = limiter.get_stats()
        print(f"\n   Rate limiter stats:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Rate limiter test failed: {e}")
        return False


def test_ai_engine():
    """Test 5: AI Engine Integration"""
    print_header("TEST 5: AI Engine Integration")

    try:
        # Create mock bot
        class MockBot:
            def log(self, msg):
                print(f"   [Bot] {msg}")

            ui_regions = {
                'region_full_screen': {'x': 0, 'y': 0, 'w': 1280, 'h': 720}
            }

            class LocationManager:
                def get_region_name(self):
                    return "EXP_FARM"

            location = LocationManager()

        bot = MockBot()

        # Initialize AI engine
        engine = AIEngine(bot, config_path="config/ai_config.yaml")
        print("✅ AI Engine initialized")

        # Check configuration
        print(f"   - Enabled: {engine.enabled}")
        print(f"   - Observer mode: {engine.observer_mode}")
        print(f"   - Fallback to rules: {engine.fallback_to_rules}")

        # Get statistics
        stats = engine.get_statistics()
        print(f"\n   AI Engine statistics:")
        print(f"   - Decisions made: {stats['decisions_made']}")
        print(f"   - AI calls: {stats['ai_calls_made']}")
        print(f"   - Fallback count: {stats['fallback_count']}")
        print(f"   - Cache hit rate: {stats.get('cache_hit_rate', 0)}%")

        return True

    except Exception as e:
        print(f"❌ AI Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_freeze_diagnosis():
    """Test 6: AI Freeze Diagnosis (Phase 2)"""
    print_header("TEST 6: AI Freeze Diagnosis (Phase 2)")

    try:
        class MockBot:
            _last_action = "enter_boss_map"
            _expected_result = "map_loaded"
            def log(self, msg):
                print(f"   [Bot] {msg}")
            ui_regions = {'region_full_screen': {'x': 0, 'y': 0, 'w': 1280, 'h': 720}}
            settings = {"FREEZE_CHECK_INTERVAL_SN": 15}
            class LocationManager:
                def get_region_name(self): return "KATMAN_1"
            location = LocationManager()

        bot = MockBot()
        engine = AIEngine(bot, config_path="config/ai_config.yaml")

        # Test that diagnose_freeze method exists and is callable
        assert hasattr(engine, 'diagnose_freeze'), "diagnose_freeze method missing"
        print("✅ diagnose_freeze() method exists")

        # Test with disabled feature flag (should return None)
        original = engine.features.get('freeze_diagnosis')
        engine.features['freeze_diagnosis'] = False
        result = engine.diagnose_freeze(duration=60.0, last_action="test", expected_result="test")
        assert result is None, "Should return None when feature disabled"
        print("✅ Returns None when feature disabled")

        # Restore
        engine.features['freeze_diagnosis'] = original
        print("✅ Freeze diagnosis feature flag works correctly")

        return True

    except Exception as e:
        print(f"❌ Phase 2 freeze diagnosis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_strategic_wait():
    """Test 7: AI Strategic Wait (Phase 2)"""
    print_header("TEST 7: AI Strategic Wait (Phase 2)")

    try:
        class MockCombat:
            def get_walk_time(self, a, b): return 40

        class MockBot:
            def log(self, msg):
                print(f"   [Bot] {msg}")
            ui_regions = {'region_full_screen': {'x': 0, 'y': 0, 'w': 1280, 'h': 720}}
            combat = MockCombat()
            class LocationManager:
                def get_region_name(self): return "KATMAN_1"
            location = LocationManager()

        bot = MockBot()
        engine = AIEngine(bot, config_path="config/ai_config.yaml")

        # Test that evaluate_strategic_wait method exists
        assert hasattr(engine, 'evaluate_strategic_wait'), "evaluate_strategic_wait method missing"
        print("✅ evaluate_strategic_wait() method exists")

        # Test with disabled feature flag
        original = engine.features.get('strategic_wait')
        engine.features['strategic_wait'] = False
        current = {'aciklama': 'boss_800', 'katman_id': 'katman_1'}
        next_b = {'aciklama': 'boss_820', 'katman_id': 'katman_1', 'spawn_time': time.time() + 60}
        result = engine.evaluate_strategic_wait(current, next_b, 60.0)
        assert result is None, "Should return None when feature disabled"
        print("✅ Returns None when feature disabled")

        # Restore
        engine.features['strategic_wait'] = original
        print("✅ Strategic wait feature flag works correctly")

        # Test cache hash generation for strategic wait
        state_hash = engine.cache.hash_state(
            type="strategic_wait",
            current="boss_800",
            next="boss_820",
            time_bucket="6",
            same_map="True"
        )
        assert len(state_hash) == 16, "Hash should be 16 chars"
        print(f"✅ State hash generated: {state_hash}")

        return True

    except Exception as e:
        print(f"❌ Phase 2 strategic wait test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_decision_logger():
    """Test 8: Decision History Logger (Phase 3)"""
    print_header("TEST 8: Decision History Logger (Phase 3)")

    try:
        from ai_engine import DecisionHistoryLogger

        logger = DecisionHistoryLogger("data/decision_history_test.jsonl")
        print("✅ DecisionHistoryLogger initialized")

        # Log some test decisions
        logger.log_decision('boss_selection',
            {'decision': 'boss_800', 'confidence': 0.85},
            {'location': 'KATMAN_1'}, source='ai')

        logger.log_decision('strategic_wait',
            {'decision': 'wait', 'confidence': 0.7},
            {'time_until_next': 45}, source='ai')

        logger.log_decision('freeze_diagnosis',
            {'decision': 'restart', 'reason': 'ai_returned_none'},
            {'duration': 60}, source='rules')

        print("✅ 3 decisions logged successfully")

        # Read back
        recent = logger.get_recent_decisions(count=10)
        assert len(recent) >= 3, f"Expected 3+ decisions, got {len(recent)}"
        print(f"✅ Read back {len(recent)} decisions")

        # Get stats
        stats = logger.get_stats()
        assert stats['total_logged'] >= 3
        assert stats['ai_decisions'] >= 2
        assert stats['fallback_decisions'] >= 1
        print(f"✅ Stats: {stats['total_logged']} total, {stats['ai_decisions']} AI, {stats['fallback_decisions']} rules")
        print(f"   AI ratio: {stats['ai_ratio']}%")

        # Filter by type
        boss_only = logger.get_recent_decisions(count=10, decision_type='boss_selection')
        assert len(boss_only) >= 1
        print(f"✅ Type filter works: {len(boss_only)} boss_selection entries")

        # Clean up test file
        import os
        try:
            os.remove("data/decision_history_test.jsonl")
        except:
            pass

        return True

    except Exception as e:
        print(f"❌ Decision logger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_analytics():
    """Test 9: Performance Analytics (Phase 3)"""
    print_header("TEST 9: Performance Analytics (Phase 3)")

    try:
        from analytics import PerformanceAnalytics

        analytics = PerformanceAnalytics(
            history_path="data/decision_history.jsonl",
            memory_path="data/ai_memory.json"
        )
        print("✅ PerformanceAnalytics initialized")

        # Get dashboard data (even if empty)
        data = analytics.get_dashboard_data()
        assert 'ai_vs_rules' in data
        assert 'boss_performance' in data
        assert 'decision_breakdown' in data
        assert 'top_bosses' in data
        print("✅ Dashboard data structure correct")

        # Get text report
        report = analytics.get_text_report()
        assert len(report) > 50
        print("✅ Text report generated")
        print(f"   Report length: {len(report)} chars")

        return True

    except Exception as e:
        print(f"❌ Analytics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_ai_toggle():
    """Test 10: AI Toggle (Phase 3)"""
    print_header("TEST 10: AI Toggle (Phase 3)")

    try:
        class MockBot:
            def log(self, msg):
                print(f"   [Bot] {msg}")
            ui_regions = {'region_full_screen': {'x': 0, 'y': 0, 'w': 1280, 'h': 720}}
            class LocationManager:
                def get_region_name(self): return "EXP_FARM"
            location = LocationManager()

        bot = MockBot()
        engine = AIEngine(bot, config_path="config/ai_config.yaml")

        # Test toggle
        original = engine.enabled
        assert original == True, "AI should start enabled"
        print(f"✅ AI starts enabled: {original}")

        engine.toggle_ai(False)
        assert engine.enabled == False
        print("✅ AI disabled via toggle_ai(False)")

        engine.toggle_ai()
        assert engine.enabled == True
        print("✅ AI re-enabled via toggle_ai() (toggle)")

        engine.toggle_ai()
        assert engine.enabled == False
        print("✅ AI toggled off again")

        # Restore
        engine.enabled = original
        print(f"✅ Restored to original state: {engine.enabled}")

        return True

    except Exception as e:
        print(f"❌ AI toggle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("#" + " "*22 + "AI SYSTEM TEST" + " "*22 + "#")
    print("#"*60)

    tests = [
        ("Configuration Loading", test_config),
        ("Gemini Client", test_gemini_client),
        ("Memory Manager", test_memory_manager),
        ("Rate Limiter", test_rate_limiter),
        ("AI Engine", test_ai_engine),
        ("Phase 2: Freeze Diagnosis", test_phase2_freeze_diagnosis),
        ("Phase 2: Strategic Wait", test_phase2_strategic_wait),
        ("Phase 3: Decision Logger", test_phase3_decision_logger),
        ("Phase 3: Analytics", test_phase3_analytics),
        ("Phase 3: AI Toggle", test_phase3_ai_toggle)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed! AI system is ready.")
        print("\n  Next steps:")
        print("  1. Run main.py to start the bot")
        print("  2. Check logs/ai_decisions.log for AI activity")
        print("  3. Monitor data/ai_memory.json for learning progress")
    else:
        print("\n  ⚠️ Some tests failed. Please fix issues before running bot.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
