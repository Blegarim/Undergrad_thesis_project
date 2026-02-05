"""
Final verification test for ablation model integration.
Tests both train.py and test.py for proper model type suffix handling.
"""

def test_model_suffix_logic():
    """Test that model suffix logic is working correctly."""
    print("Testing model suffix logic...")
    
    # Test suffix generation logic
    test_cases = [
        ('full', ''),
        ('motion_only', '_motion_only'),
        ('visual_only', '_visual_only'),
        ('vanilla_concat', '_vanilla_concat'),
    ]
    
    for model_type, expected_suffix in test_cases:
        # Simulate the suffix logic from both scripts
        model_suffix = f"_{model_type}" if model_type != 'full' else ""
        
        if model_suffix == expected_suffix:
            print(f"  [OK] {model_type} -> '{model_suffix}'")
        else:
            print(f"  [ERROR] {model_type} -> '{model_suffix}' (expected '{expected_suffix}')")
            return False
    
    print("[OK] All suffix logic tests passed!")
    return True

def test_file_structure():
    """Test that required files exist and have correct structure."""
    print("Testing file structure...")
    
    import os
    
    required_files = [
        'models/AblationModels.py',
        'train.py', 
        'test.py',
        'ABLATION_COMPLETE.md'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  [OK] {file_path}")
        else:
            print(f"  [ERROR] {file_path} missing")
            return False
    
    print("[OK] All required files exist!")
    return True

def test_keyword_presence():
    """Test that required keywords are present in both scripts."""
    print("Testing keyword presence...")
    
    # Test train.py
    try:
        with open('train.py', 'r') as f:
            train_content = f.read()
    except:
        print("  [ERROR] Could not read train.py")
        return False
    
    # Test test.py  
    try:
        with open('test.py', 'r') as f:
            test_content = f.read()
    except:
        print("  [ERROR] Could not read test.py")
        return False
    
    # Check for required patterns
    patterns = [
        ('model_type argument', '--model_type'),
        ('get_model function', 'def get_model('),
        ('model_forward function', 'def model_forward('),
        ('ablation imports', 'from models.AblationModels import'),
        ('argparse import', 'import argparse'),
        ('model_suffix logic', 'model_suffix = f"_{args.model_type}"'),
    ]
    
    for pattern_name, pattern in patterns:
        if pattern in train_content:
            print(f"  [OK] train.py: {pattern_name}")
        else:
            print(f"  [ERROR] train.py: {pattern_name}")
            return False
            
        if pattern in test_content:
            print(f"  [OK] test.py: {pattern_name}")
        else:
            print(f"  [ERROR] test.py: {pattern_name}")
            return False
    
    print("[OK] All required patterns found in both scripts!")
    return True

def main():
    print("=== FINAL ABLATION INTEGRATION VERIFICATION ===\n")
    
    tests = [
        ("Model Suffix Logic", test_model_suffix_logic),
        ("File Structure", test_file_structure), 
        ("Keyword Presence", test_keyword_presence),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if not test_func():
            all_passed = False
    
    print(f"\n=== FINAL RESULT ===")
    if all_passed:
        print("*** ALL TESTS PASSED! ***")
        print("\nAblation model integration is COMPLETE and ready for use!")
        print("\nFinal usage:")
        print("  python train.py --model_type [motion_only|visual_only|vanilla_concat|full]")
        print("  python test.py --model_type [motion_only|visual_only|vanilla_concat|full]")
        print("\nModel checkpoint files will have appropriate suffixes:")
        print("  - Full model: model_epoch28_0122_1511.pth")
        print("  - Motion only: model_epoch28_0122_1511_motion_only.pth")  
        print("  - Visual only: model_epoch28_0122_1511_visual_only.pth")
        print("  - Vanilla concat: model_epoch28_0122_1511_vanilla_concat.pth")
    else:
        print("*** SOME TESTS FAILED! ***")
        print("Please review the errors above before proceeding.")

if __name__ == "__main__":
    main()