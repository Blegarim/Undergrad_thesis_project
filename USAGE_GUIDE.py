"""
Step-by-step usage guide for class imbalance handling.
"""

print("""
=== STEP-BY-STEP USAGE GUIDE ===

STEP 1: QUICK VALIDATION (Already Done ✅)
- Ran basic_test.py - PASSED
- Configuration files working correctly
- All presets available

STEP 2: CHOOSE YOUR STRATEGY

Option A: RECOMMENDED (Start Here)
Command: python train_enhanced.py --preset recommended --enable_advanced True
Risk: Low | Expected Improvement: Medium-High
Best for: First time users, balanced approach

Option B: CONSERVATIVE  
Command: python train_enhanced.py --preset conservative --enable_advanced True
Risk: Very Low | Expected Improvement: Low-Medium  
Best for: Testing, minimal changes to training dynamics

Option C: AGGRESSIVE
Command: python train_enhanced.py --preset aggressive --enable_advanced True
Risk: Medium | Expected Improvement: High
Best for: Maximum performance, willing to tune parameters

Option D: ORIGINAL (No Changes)
Command: python train.py
Risk: None | Expected Improvement: None
Best for: Baseline comparison

STEP 3: MONITORING

When you run enhanced training, watch for:
- F1-score improvements (not just accuracy)
- Loss component breakdown per task  
- Dynamic weight adjustments
- Better minority class performance

Log files will be saved as:
training_log/training_log_enhanced_MMDD_HHMM.csv

Enhanced metrics include:
- Actions/Looks/Crosses: Accuracy, F1, Precision, Recall
- Macro F1 for overall performance
- Better early stopping based on F1-score

STEP 4: COMPARISON

After running enhanced training:
1. Compare F1-scores with your original training logs
2. Focus on Looks and Crosses tasks (should see biggest improvements)
3. Actions task should maintain or slightly improve
4. Overall macro F1 should increase

STEP 5: TUNING (If Needed)

If results aren't as expected:
- Try conservative preset first
- Gradually increase to recommended
- Only use aggressive if you're comfortable debugging
- Monitor training stability

EXPECTED IMPROVEMENTS:

Looks Task (17% positive, 83% negative):
- Current F1: ~0.1-0.2
- Expected F1: ~0.3-0.5 (with recommended preset)

Crosses Task (2% positive, 98% negative):  
- Current F1: ~0.2-0.3
- Expected F1: ~0.4-0.6 (with recommended preset)

Actions Task (52% positive, 48% negative):
- Current F1: ~0.4-0.6
- Expected F1: ~0.5-0.7 (maintained or slight improvement)

STEP 6: PRODUCTION DEPLOYMENT

Once satisfied with results:
1. Use enhanced training for final model
2. Monitor inference performance on validation set
3. Compare with baseline model
4. Deploy the better performing model

KEY BENEFITS:

✅ Task-specific solutions for different imbalance levels
✅ Mathematically grounded loss functions  
✅ Dynamic adaptation during training
✅ Better evaluation metrics (F1 > accuracy)
✅ Safe, toggleable implementation
✅ Zero changes to original pipeline
✅ Preset configurations for easy use

NEXT STEPS:

1. Start with: python train_enhanced.py --preset recommended --enable_advanced True
2. Monitor training logs for improvements
3. Compare with baseline performance
4. Adjust preset if needed

Your setup is ready to address severe class imbalance effectively!
""")