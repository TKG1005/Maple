#!/usr/bin/env python3
"""Test script for performance profiling system."""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

def test_basic_profiling():
    """Test basic profiling functionality."""
    print("Testing basic profiling functionality...")
    
    # Test basic profiling
    python_cmd = "python train.py --episodes 1 --parallel 1 --profile --profile-name test_basic"
    print(f"Running: {python_cmd}")
    
    import subprocess
    result = subprocess.run(python_cmd.split(), capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Basic profiling test successful")
        return True
    else:
        print("❌ Basic profiling test failed")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        print("STDERR:", result.stderr[-500:])  # Last 500 chars
        return False

def test_cross_platform_profiling():
    """Test profiling across different configurations."""
    print("\nTesting cross-platform profiling...")
    
    test_configs = [
        {
            "name": "cpu_single",
            "cmd": "python train.py --episodes 1 --parallel 1 --device cpu --profile --profile-name cpu_single"
        },
        {
            "name": "cpu_multi", 
            "cmd": "python train.py --episodes 1 --parallel 3 --device cpu --profile --profile-name cpu_multi"
        },
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"Testing {config['name']}...")
        import subprocess
        result = subprocess.run(config['cmd'].split(), capture_output=True, text=True)
        results[config['name']] = result.returncode == 0
        
        if results[config['name']]:
            print(f"✅ {config['name']} test successful")
        else:
            print(f"❌ {config['name']} test failed")
            print("STDERR:", result.stderr[-200:])
    
    return all(results.values())

def check_output_files():
    """Check if profiling output files were created."""
    print("\nChecking profiling output files...")
    
    profiling_dir = Path("logs/profiling")
    
    if not profiling_dir.exists():
        print("❌ Profiling directory not found")
        return False
    
    # Check for raw profiling data
    raw_dir = profiling_dir / "raw"
    if raw_dir.exists():
        json_files = list(raw_dir.glob("*.json"))
        print(f"✅ Found {len(json_files)} raw profiling files")
    else:
        print("❌ No raw profiling directory found")
        return False
    
    # Check for reports
    reports_dir = profiling_dir / "reports"
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.txt"))
        print(f"✅ Found {len(report_files)} profiling reports")
    else:
        print("❌ No reports directory found")
        return False
    
    # Show latest file
    if json_files:
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📄 Latest profiling file: {latest_file.name}")
        
        # Show file size
        size_kb = latest_file.stat().st_size / 1024
        print(f"📊 File size: {size_kb:.1f} KB")
    
    return True

def main():
    """Run all profiling tests."""
    print("🚀 Starting Performance Profiling System Tests")
    print("=" * 50)
    
    # Test 1: Basic functionality
    test1_result = test_basic_profiling()
    
    # Test 2: Cross-platform configurations  
    test2_result = test_cross_platform_profiling()
    
    # Test 3: Output file verification
    test3_result = check_output_files()
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"Basic Profiling: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Cross-Platform: {'✅ PASS' if test2_result else '❌ FAIL'}")
    print(f"Output Files: {'✅ PASS' if test3_result else '❌ FAIL'}")
    
    overall_result = all([test1_result, test2_result, test3_result])
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_result else '❌ SOME TESTS FAILED'}")
    
    if overall_result:
        print("\n🎉 Performance profiling system is working correctly!")
        print("💡 Usage examples:")
        print("  python train.py --episodes 10 --profile --profile-name my_experiment")
        print("  python train.py --episodes 50 --parallel 10 --device cpu --profile")
        print(f"📁 Check results in: {Path('logs/profiling').absolute()}")
    
    return overall_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)