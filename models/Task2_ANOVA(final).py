import pandas as pd
from scipy.stats import f_oneway

# Load dataset
df = pd.read_csv("student_data.csv")

print("Dataset loaded successfully!")
print(df.head())

# ================================
# Split ONLY 3 columns into groups
# ================================

# Study hours groups
df["study_group"] = pd.qcut(df["study_hours"], q=3, labels=["Low", "Medium", "High"])

# Class attendance groups
df["attendance_group"] = pd.qcut(df["class_attendance"], q=3, labels=["Low", "Medium", "High"])

# Sleep hours groups
df["sleep_group"] = pd.qcut(df["sleep_hours"], q=3, labels=["Low", "Medium", "High"])


# =========================================
# Function to run ANOVA for a grouped column
# =========================================
def run_anova(group_column, name):

    groups = []
    labels = ["Low", "Medium", "High"]

    for label in labels:
        group = df[df[group_column] == label]["exam_score"]
        groups.append(group)

    print(f"\n{name} Group Sizes:")
    for label, g in zip(labels, groups):
        print(f"{label}: {len(g)}")

    # ANOVA test
    f_stat, p_value = f_oneway(*groups)

    print(f"\nANOVA Result for {name}:")
    print("F-statistic:", f_stat)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Result: There IS a significant difference between groups.")
    else:
        print("Result: There is NO significant difference between groups.")


# ======================
# Run ANOVA for 3 factors
# ======================

run_anova("study_group", "Study Hours")
run_anova("attendance_group", "Class Attendance")
run_anova("sleep_group", "Sleep Hours")