#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### 1. Load Dataset
# This project uses the NYC Building Energy and Water Disclosure dataset for calendar years 2022 to present. 
# Since all team members use the same dataset but study different subproblems, this shared EDA establishes a common understanding of the dataset structure, data quality, and shared preprocessing rules.

# In[4]:


file_path = "NYC_Building_Energy_and_Water_Data_Disclosure_for_Local_Law_84_2023_to_Present_(Data_for_Calendar_Year_2022-Present)_20260329.csv"
df = pd.read_csv(file_path, low_memory=False)


# In[6]:


print("Shape:", df.shape)


# In[8]:


df.head()


# In[10]:


print("\nColumn sample:")
print(df.columns.tolist()[:30])


# ### 2. Data Types and Rough Column Categories

# In[13]:


dtype_summary = df.dtypes.value_counts()
dtype_summary


# In[15]:


num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", len(num_cols))
print("Non-numeric columns:", len(cat_cols))


# **Takeaway:** The dataset is dominated by non-numeric variables, so type conversion and selective feature filtering will be important in later task-specific analysis.

# In[17]:


building_cols = [c for c in df.columns if any(k in c.lower() for k in [
      "year built", "property type", "gfa", "floor area", "number of buildings", "city", "borough"
  ])]

energy_cols = [c for c in df.columns if any(k in c.lower() for k in [
      "energy", "eui", "electricity", "natural gas", "fuel oil", "steam"
  ])]
water_cols = [c for c in df.columns if "water" in c.lower()]

emissions_cols = [c for c in df.columns if any(k in c.lower() for k in [
      "ghg", "emissions", "co2"
  ])]

id_meta_cols = [c for c in df.columns if any(k in c.lower() for k in [
      "id", "name", "address", "postal", "bbl", "bin", "year ending", "calendar year"
  ])]


# In[19]:


print("Building-related columns:",
len(building_cols))
print("Energy-related columns:",
len(energy_cols))
print("Water-related columns:", len(water_cols))
print("Emissions-related columns:",
len(emissions_cols))
print("ID / metadata columns:",
len(id_meta_cols))


# In[21]:


print("Sample building columns:",
building_cols[:10])
print("Sample energy columns:", energy_cols[:10])
print("Sample water columns:", water_cols[:10])
print("Sample emissions columns:",
emissions_cols[:10])


# **Takeaway:** These are rough keyword-based groupings for dataset overview only, not final feature selection.

# ### 3. Data Inspection

# In[59]:


# Check placeholder values in raw data
placeholder = "Not Available"

not_available_counts = (df.astype(str) == placeholder).sum().sort_values(ascending=False)
not_available_counts = not_available_counts[not_available_counts > 0]

not_available_counts.head(20)


# In[63]:


if len(not_available_counts) > 0:
  plt.figure(figsize=(10, 6))

  not_available_counts.head(20).sort_values().plot(kind="barh")
  plt.title('Top 20 Columns Containing "Not Available"')
  plt.xlabel("Count")
  plt.tight_layout()
  plt.show()


# **Takeaway:** The raw data uses `Not Available` as a placeholder in many columns. This means true missingness is understated in the original dataset and should be reassessed after standardizing these placeholders.

# In[68]:


# Create shared-cleaned data
df_shared = df.replace("Not Available", np.nan).infer_objects(copy=False).copy()


# In[70]:


# Missingness after shared cleaning
missing_summary = pd.DataFrame({
  "missing_count": df_shared.isna().sum(),
  "missing_rate": df_shared.isna().mean()
}).sort_values("missing_rate", ascending=False)

missing_summary.head(20)


# **Takeaway:** After converting `Not Available` to missing values, several columns become extremely sparse or nearly empty. This suggests that the dataset contains many domain-specific fields that only apply to a small subset of properties, so task-specific column filtering will be necessary later.

# In[73]:


# Duplicate check
duplicate_count = df_shared.duplicated().sum()
print("Duplicate rows:", duplicate_count)


# **Takeaway:** No fully duplicated rows were found in the shared-cleaned dataset, so duplicate removal does not appear to be a major issue at the dataset level.

# ### 4. Descriptive Statistics

# In[76]:


shared_numeric = [
  "Year Built",
  "Property GFA - Self-Reported (ft²)",
  "Number of Buildings"
]


# In[78]:


for col in shared_numeric:
  if col in df_shared.columns:
      df_shared[col] = pd.to_numeric(df_shared[col].astype(str).str.replace(",", "", regex=False), errors="coerce")


# In[80]:


# Descriptive statistics for numeric variables
numeric_existing = [c for c in shared_numeric if c in df_shared.columns]

numeric_summary = df_shared[numeric_existing].describe().T
numeric_summary["missing_count"] = df_shared[numeric_existing].isna().sum()
numeric_summary["missing_rate"] = df_shared[numeric_existing].isna().mean()

numeric_summary


# In[49]:


shared_categorical = [
  "Primary Property Type - Self Selected"
]


# In[51]:


# Descriptive statistics for categorical variable
cat_col = "Primary Property Type - Self Selected"

type_summary = pd.DataFrame({
  "count": df_shared[cat_col].value_counts(dropna=False),
  "proportion": df_shared[cat_col].value_counts(dropna=False, normalize=True)
})

print("Number of unique categories:", df_shared[cat_col].nunique(dropna=True))
type_summary.head(15)


# **Takeaway:** The shared numerical variables vary substantially in scale, and Number of Buildings is concentrated at small values with a long right tail.

# ### 5. Distribution Checks

# In[54]:


# Distribution checks for numeric variables 
for col in numeric_existing:
  series = df_shared[col].dropna()

  plt.figure(figsize=(6, 4))
  sns.histplot(series, bins=40, kde=False)
  plt.title(f"Histogram: {col}")
  plt.xlabel(col)
  plt.ylabel("Count")
  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(6, 2.5))
  sns.boxplot(x=series)
  plt.title(f"Boxplot: {col}")
  plt.xlabel(col)
  plt.tight_layout()
  plt.show()


# In[56]:


# Distribution checks for categorical values
plt.figure(figsize=(10, 6))
df_shared[cat_col].value_counts().head(15).sort_values().plot(kind="barh")
plt.title("Top 15 Primary Property Types")
plt.xlabel("Count")
plt.ylabel("Property Type")
plt.tight_layout()
plt.show()


# **Takeaway:** The plots confirm skewness and possible extreme values in shared numeric variables, especially Property GFA and Number of Buildings, while property types are strongly imbalanced.

# ### 6. Shared Cleaning Rules
# Based on the shared EDA, the team will use the following common preprocessing rules:
#   1. Replace placeholder values such as `Not Available` with missing values (`NaN`).
#   2. Reassess missingness after placeholder conversion because raw null counts alone understate data sparsity.
#   3. Apply basic numeric type conversion to common shared-analysis variables such as `Year Built`, `Property GFA - Self-Reported (ft²)`, and `Number of Buildings`.
#   4. Do not use identifier-like columns as modeling features unless they are specifically needed for grouping or interpretation.
#   5. Keep the shared EDA limited to common data understanding, and perform target-specific preprocessing only in each member's own task notebook.

# ### 7. Shared Findings
# The shared EDA shows that this dataset is rich enough to support multiple machine learning tasks, including regression, classification, and clustering, because it contains building characteristics, energy indicators, water indicators, and emissions-related variables.
# 
# At the same time, the dataset requires careful preprocessing. A large portion of the columns are non-numeric, many fields use `Not Available` as a placeholder, and several columns become highly sparse after placeholder standardization. This indicates that the dataset is wide and heterogeneous, and many variables only apply to a subset of properties.
# 
# The shared numerical variables also show clear distribution issues. `Number of Buildings` is highly right-skewed, and `Year Built` includes potentially suspicious values that may need task-specific review later. The categorical distribution is also imbalanced, with `Multifamily Housing` accounting for about 66.5% of the records.
# 
# Based on these findings, the team will use a common preprocessing approach at the shared level: standardize placeholder values, inspect missingness after cleaning, and treat only a small set of common variables as shared analysis variables. After this shared EDA, each team member will continue with task-specific EDA using their own target variable and feature subset.

# ### 8. Export Shared-Cleaned Dataset
# This shared-cleaned dataset is exported as the common starting point for all team members' individual EDA and preprocessing. It includes placeholder standardization (`Not Available` to `NaN`) but does not apply any task-specific filtering, imputation, or encoding.

# In[106]:


output_path = "NYC_Building_Energy_and_Water_Data_Disclosure_shared_cleaned.csv"
df_shared.to_csv(output_path, index=False)

print("Exported shared-cleaned dataset to:", output_path)

