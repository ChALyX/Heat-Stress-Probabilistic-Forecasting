# COMP0197: Applied Deep Learning
## Assessed Component 2 (Group Work, in-person – 25%)
## Assessed Component 3 (Individual Report – 25%)
*Submission on Moodle*

---

## Group Project: Collaborative Research and Development

### 1. Project Overview and Scope

The group project (50% of the module) requires groups of students to design, implement and evaluate a deep learning system for a complex problem of their choosing. This project assesses your ability to reason about design choices, implement frameworks and communicate findings in research-style report and presentation.

- **Technical Scope**: Develop a system capable of predicting future values in a complex temporal sequence while quantifying the data (aleatoric) and/or model (epistemic) uncertainty associated with those predictions.
- **Core Requirement**: You must move beyond standard deterministic forecasting. Your model must treat the output not as a single value, but as a probability distribution.
- **Open Design**: This is an open project. Students are free to define their own motivations, select their application domain and design an original implementation within the defined scope of sequential uncertainty.
- **Data Choice**: Groups may choose any sequential dataset (e.g., global weather patterns, energy consumption, or physiological signals).
- **Environment**: All code must be compatible with one of the micromamba environments used in Coursework 1 (`comp0197-pt` or `comp0197-tf`). You must use only one framework for the entire project.

---

### 2. Assessed Component 2: Progress Presentation and Design Defence (25%)

This is an in-person technical audit where the group presents the project's conceptual design and current implementation status. No code is submitted at this stage, but the group must demonstrate (and defend through Q&As) a clear path to a working system.

**Key Discussion Points:**

- **Design and Rationale**: Justification for the selected problem, dataset choice and the specific high-level architecture used.
- **Implementation Plan**: A roadmap of how the system components (data loader, model, training loop and evaluation) are being integrated.
- **Current Progress**: Demonstration of functional code segments, such as a working data pipeline or initial training benchmarks.
- **Problem Solving**: How the group has addressed technical hurdles like data imbalance, slow convergence or hardware limitations.

---

### 3. Assessed Component 3: Final Submission & Individual Report (25%)

This final component requires the submission of a robust, working system and an analytical individual report detailing the group's findings.

#### 3.1 Code Submission

The group must submit a standalone, reproducible codebase:

- **`train.py`**: A script that automates data retrieval, trains (or fine-tunes) the models, and saves the final weights.
- **`test.py`**: A script that loads the saved models to produce final metrics and visual results.
- **Saved Models**: The final trained model files must be included in the submission folder.
- **`instruction.pdf`**: Must list additional installed packages (max 3) and provide detailed steps to reproduce all reported results.

#### 3.2 Individual Report

The report must follow the **LNCS template** and not exceed **8 pages total** (excluding references).

**Part A: Group Technical Report (80% / Max 6 pages)**
This section is identical for all group members and should be structured like a research paper.

- **Title**: An informative title reflecting your specific study.
- **Introduction**: Define the background and literature. Clearly state your motivation and the specific problem your system addresses.
- **Methods**: Technical details of the implemented algorithms. Explain your choice of sequential model and the mathematical approach used to output a probability distribution (e.g., Gaussian likelihood, Dropout-as-inference, etc.).
- **Experiments**: Describe your experimental setup. Include ablation studies (e.g., testing the model with and without specific components) and comparisons to deterministic baselines.
- **Results**: Provide a quantitative analysis using appropriate metrics. Use clear, illustrative figures and tables.
- **Discussion**: Interpret key findings. Address unanswered questions, limitations of your current approach, and potential future directions.

**Part B: Individual Reflection and Criticism (20% / Max 2 pages)**
This section must be written independently by each student.

- **Personal Contribution**: A summary of your specific role and contribution in building the "working system" (e.g., data engineering, training optimization or visualization).
- **Critical Evaluation**: A technical critique of the final system. Identify specific weaknesses (e.g., edge cases where the model fails) and discuss how you would improve the system's robustness or efficiency if you had more time.
- **GenAI Audit**: An evidenced assessment of how GenAI tools were used and how you personally verified the output for technical accuracy.

---

### 4. Marking Criteria

Marking is primarily based on the report, based on:

- **Scientific Soundness**: Reasoning and justification of problems, methods and experiments.
- **Technical Accuracy**: Correct use of terminology, methodology, data and code.
- **Completeness**: Objective achieved and completeness of the report.
- **Presentation**: Writing organization, clarity and code readability.
- **Critical Appraisal**: Conclusive results and informative analysis.

---

> Generative AI (GenAI) tools can be used in an assistive role in this coursework, in accordance with the UCL regulations. If used, a statement must be included in each submitted Python file detailing their role in the coursework. It is your responsibility to ensure the code runs following the provided instruction.
