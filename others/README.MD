**Note**

This solution is based on modifications and additions to our model, inspired by the second-place solution in the competition.

You can find the full solution at [here](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/2nd-place-solution)

---

**Improvement in Public/Private LB and CV**

| method               | OOF CV | Public LB | Private LB |
| -------------------- | ------ | --------- | ---------- |
| No post-processing   | 0.8558 | 0.837     | 0.823      |
| with post-processing | 0.8558 | 0.875     | 0.852      |

---

**Post-processing Description**  

- We apply joint label assignment per subject under a no-repetition constraint by solving an assignment problem with the Hungarian algorithm on negative log-probabilities. 

---

**Files Included:**
- `lb0.861-102target.py`: Training script that implements the model architecture, training loop, and cross-validation strategy. 
- `inference notebook.ipynb`: Jupyter notebook for inference. Remember to update the model and data paths before running.

---

**How to Reproduce the Results:**

1. Run `lb0.861-102target.py` to train the model.
2. Execute `inference notebook.ipynb` to generate test predictions.
3. Ensure post-processing is applied during inference.
4. Submit the resulting CSV to Kaggle for scoring.

---

**Acknowledgments**  
Thank the author of the 2nd place solution for sharing their approach, which inspired our post-processing strategy and model improvements.