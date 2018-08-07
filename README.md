# qa4ie

### qa4iecode (qa4ie original code)

### scoringcode (scoring model code)
Model based on BIDAF
1. Use basic/preproscr.py to preprocess the original dataset.
2. Run basic/cli.py train the model and get inference result.
3. Use basic/prodata.py and basic/preproqa.py to get qa4ie dataset with compressed context sentences.

### scoreqacode (qa4ie with sentence selection)
Similar to qa4iecode, with data interface modified.
Get new dataset through the method in scoringcode above. Run model as qa4iecode.
