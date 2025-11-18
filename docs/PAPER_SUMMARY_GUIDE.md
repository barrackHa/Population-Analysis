# Scientific Paper Summary Guide

This guide provides a structured template for reading and summarizing scientific papers, particularly those related to neural population analysis and the stop-signal task.

---

## Summary Template

### 1. Paper Metadata

**Title**: [Full paper title]

**Journal**: [Journal name]

**Year**: [Year of publication]

**Authors**: [First Author et al.] OR [First Author & Second Author] (if only 2 authors)

---

### 2. Core Questions (up to 3 lines each)

**Why is this paper important?**
[Answer in up to 3 lines]

**What is the paper about and what is its main take-home message?**
[Answer in up to 3 lines]

**How did they get to that?**
[Answer in up to 3 lines - brief overview of approach]

**So what about it?**
[Answer in up to 3 lines - implications and significance]

---

### 3. Experimental Paradigm

[Describe the experimental setup, task structure, subjects, and general approach]

---

### 4. Results

List each major result with a brief description (up to 3 sentences per result):

**Result 1**: [Title or brief description]
[Up to 3 sentences describing the result]

**Result 2**: [Title or brief description]
[Up to 3 sentences describing the result]

[Continue as needed...]

---

### 5. Methods

[Describe the main methodological approaches used in the study. Include:
- Recording techniques
- Analysis methods
- Statistical approaches
- Key parameters or preprocessing steps]

---

### 6. Figures

**Figure 1**: [Brief description of what the figure shows]

**Figure 2**: [Brief description of what the figure shows]

[Continue for all figures...]

---

### 7. PCA/Dimensionality Reduction Analysis (if applicable)

**Data Matrix Construction**:
[Describe how the data matrix was built. Typically:
- Dimensions: N × (C × T) where N = neurons, C = conditions, T = time bins
- Alignment points used
- Epoch selection
- Trial averaging approach
- Any normalization or preprocessing]

**PCA Execution**:
[Describe:
- How PCA was applied
- Number of components retained
- Variance explained
- Cross-validation approach (if any)
- Trajectory analysis methods]

**Key PCA Findings**:
[Summarize the main findings from the PCA analysis]

---

## Important Guidelines

1. **Be Concise and Exact**: Only write information that can be directly found in the paper
2. **Cite Sources**: When the paper references another work, mention the cited paper by first author and year
3. **Accuracy Over Interpretation**: Stick to what the authors stated, avoid adding interpretations
4. **Precision in Methods**: For PCA papers, be particularly precise about matrix dimensions and construction
5. **Figure Descriptions**: Keep figure descriptions brief but informative enough to understand their purpose

---

## Usage Notes

- Use this template when creating summary files for papers in `data/papers/`
- Each paper should get its own README or summary file following this structure
- Focus on extracting actionable information relevant to our population analysis work
- Pay special attention to methodological details that might inform our own analyses

---

**Last Updated**: November 2025
