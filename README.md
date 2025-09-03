![Project Header](https://github.com/mayank1ahuja/da_ufc/blob/cd7636fe86d984715c6f2dcf15051eea842bc89c/project%20images/project%20header.png)

<h1 align = "center"><b> ✨ UFC Query Lab: SQL Driven Visuals ✨ </b></h1>

## **Project Overview**
This repository contains a reproducible, end-to-end exploratory analysis using PostgreSQL, pgAdmin4, Python and it's specialized libraries. The analysis ingests multiple CSV files, performs deterministic cleaning and feature engineering, persists cleaned tables to SQL using psycopg2, executes reproducible SQL EDA, and produces matched visualizations using both Seaborn and Plotly to verify and present results.


## **Project Goals**
- Load multiple CSV inputs into a single clean analytical dataset.
- Implement defensible cleaning rules for dates, numeric fields, and categorical values; document all decisions in the notebook.
- Engineer features that expose signal useful for descriptive analysis.
- Use SQL as the primary reporting layer, demonstrating practical SQL proficiency.
- Produce both static and interactive visualizations of the *same* SQL results: Seaborn for publication-ready PNGs; Plotly for interactive demos.
- Ensure reproducibility.

## **Dataset**
- *Source:* [Kaggle](https://www.kaggle.com) 
- *Download:* [Dataset](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)

## **Definitions**
### What is **Feature Engineering**?
Feature engineering is the deliberate creation or transformation of variables to surface signal in the data. Examples in this notebook include extracting year/month from dates, computing fighter age from birth date, and extracting weight classes.

### What is **Plotly**?
Plotly is a Python visualization library that produces **interactive, browser-ready** charts. Plotly charts support hover text, zoom/pan, clickable legends, and easy export to HTML. They are ideal for exploratory data analysis and live demos where viewers interact with the visualization to probe details.

# **Workflow**
## **Step 0: Loading Dependencies**  
Set up a reproducible Python environment and import deterministic libraries:
```python
#core dependencies
import pandas as pd
import numpy as np

#visualisation dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
```

## **Step 1: Data Ingestion**  
- Read each CSV with ```pd.read_csv()``` while appling safe flags like ```encoding_errors='ignore'``` and ```on_bad_lines='skip'``` to avoid any errors while loading the data.
```python
fighters = pd.read_csv("fighter_details.csv",
                        encoding_errors = 'ignore', 
                        on_bad_lines = 'skip') 

fights = pd.read_csv("fight_details.csv",
                        encoding_errors = 'ignore', 
                        on_bad_lines = 'skip') 

events = pd.read_csv("event_details.csv",
                        encoding_errors = 'ignore', 
                        on_bad_lines = 'skip')                       
```

### *Step 1(a): Initial Data Exploration* 
Initial data exploration is the first glance at a dataset to understand its structure, size, and basic patterns. It helps identify potential issues like missing values or inconsistent entries early on, and builds the intuition needed to guide cleaning, feature engineering, and deeper analysis.
```python
fighters.head()
fighters.shape
``` 


## **Step 2: Data Cleaning**  
Data cleaning ensures that the dataset is accurate, consistent, and analysis-ready by handling missing values, correcting data types, removing duplicates, and standardizing formats. This step is crucial because reliable insights can only come from well-prepared, trustworthy data.

### *Step 2(a): Treating Null Values* 
```python
#treating null values in Fighters
fighters['nick_name'] = fighters['nick_name'].fillna('Unknown')
fighters['stance'] = fighters['stance'].fillna('Unknown')
fighters["dob"] = fighters["dob"].fillna(pd.NaT)  # pd.NaT = missing datetime
```

### *Step 2(b): Checking for Duplicate Values* 
```python
#checking duplicates in Fighters
fighters.duplicated().sum()
```

### *Step 2(c): Treating Data Types: Typecasting*
```python
#typecasting in Fighters
fighters['dob'] = pd.to_datetime(fighters['dob'])
fighters['stance'] = fighters['stance'].astype('category')

#typecasting in Events
events['date'] = pd.to_datetime(events['date'])
```

## **Step 3: Feature Engineering**  
Feature engineering is the process of creating new, meaningful variables from raw data to better capture underlying patterns. It enhances the dataset’s predictive power and analytical depth, enabling more insightful exploration and stronger models.

### *Step 3(a): Deriving the  Weightclasses of the Fighters*
```python
def classify_weight_class(weight):
    if weight <= 56.7:
        return "Flyweight"
    elif weight <= 61.2:
        return "Bantamweight"
    elif weight <= 65.8:
        return "Featherweight"
    elif weight <= 70.3:
        return "Lightweight"
    elif weight <= 77.1:
        return "Welterweight"
    elif weight <= 83.9:
        return "Middleweight"
    elif weight <= 93.0:
        return "Light Heavyweight"
    elif weight <= 120.2:
        return "Heavyweight"
    else:
        return "Super Heavyweight"

fighters["weight_class"] = fighters["weight"].apply(classify_weight_class)
```

### *Step 3(b): Determining the Current Age of Fighters*
```python
fighters['age'] = 2025 - fighters['dob'].dt.year
```

### *Step 3(c): Extracting Year and Month from the Event Date*
```python
events["event_year"] = events["date"].dt.year
events["event_month"] = events["date"].dt.month
```

### *Step 3(d): Determining Win-Ratios of the Fighters*
```python
fighters['win_ratio'] = fighters['wins'] / (fighters['wins'] + fighters['losses'])
```

### *Step 3(e): Determining the Experience of the Fighters*
```python
fighters["experience"] = fighters["wins"] + fighters["losses"] + fighters["draws"]
```

## **Step 4: Exporting Cleaned Data**
```python 
fighters.to_csv("cleaned_fighters.csv", index=False)
events.to_csv("cleaned_events.csv", index=False)
fights.to_csv("cleaned_fights.csv", index=False)  
```

## **Step 5: Connecting to PostgreSQL**
```python
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    port=5432(default port),
    dbname="db_name",
    user="username",
    password="password"
)
```

## **Step 6: Exploratory Data Analysis(EDA)**
SQL-based EDA uses queries to summarize, filter, and aggregate data directly in the database. It allows efficient handling of large datasets, ensures reproducibility, and provides a reliable **reference point** for insights before visualization.

1. Most Common Finishes 
```sql
-- Question 1 : What are the most common finish methods and how many fights ended by each method?
SELECT method, 
	   COUNT(*) AS count
FROM fights
GROUP BY method
ORDER BY count DESC
LIMIT 20;
```

2. Distribution of Fighters by Weight Class
```sql
-- Question 2 : What is the distribution of fighters by weight class?
SELECT COALESCE(weight_class, 'Unknown') AS weight_class,
       COUNT(*) AS n_fighters
FROM fighters
GROUP BY COALESCE(weight_class, 'Unknown')
ORDER BY n_fighters DESC;
```

3. Top Fighters by Win Ratio
```sql
-- Question 3 : Who are the top fighters by win_ratio?
SELECT id, 
	   name, 
	   experience,
	   win_ratio
FROM fighters
WHERE COALESCE(experience,0) >= 5
ORDER BY win_ratio DESC
LIMIT 20;
```

4. Number of Fights per Year
```sql
-- Question 4 : What are the number of fights per year?
SELECT EXTRACT(YEAR FROM e.date)::INT AS year,
       COUNT(*) AS num_fights
FROM fights f
JOIN events e ON f.event_id = e.event_id
GROUP BY year
ORDER BY year;
```

5. Count of Finish Method
```sql
-- Question 5 : What are the total counts of each finishing method and what percentage are they of the total?
SELECT COALESCE(method,'Unknown') AS method,
       COUNT(*) AS cnt,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fights), 2) AS pct_of_all_fights
FROM fights
GROUP BY COALESCE(method,'Unknown')
ORDER BY cnt DESC;
```

6. Win Ratios by Weight Class
```sql
-- Question 6 : Define the win ratios by weight class.
SELECT weight_class,
	   COUNT(*) AS n_fighters,
	   ROUND(AVG(win_ratio)::numeric, 3) AS avg_win_ratio
FROM fighters
GROUP BY weight_class
HAVING COUNT(*) >= 5
ORDER BY avg_win_ratio DESC;
```

7. Fights per Year with a 3-Year Moving Average
```sql
-- Question 7 : How many fights happened each year, and what’s the 3-year moving average?
WITH per_year AS (
  SELECT DATE_TRUNC('year', e.date)::date AS year,
         COUNT(*) AS num_fights
  FROM fights f
  JOIN events e ON f.event_id = e.event_id
  GROUP BY year
)
SELECT year,
       num_fights,
       ROUND(AVG(num_fights) OVER (ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric,2) AS ma_3yr
FROM per_year
ORDER BY year;
```

8. Top Active Fighters per Year
```sql
-- Question 8 : Which are the top active fighters per year?
WITH participants AS (
  SELECT f.fight_id, e.date, f.r_id AS fighter_id FROM fights f JOIN events e USING(event_id)
  UNION ALL
  SELECT f.fight_id, e.date, f.b_id AS fighter_id FROM fights f JOIN events e USING(event_id)
),
per_year AS (
  SELECT DATE_TRUNC('year', date)::DATE AS year, fighter_id, COUNT(*) AS fights_count
  FROM participants
  GROUP BY year, fighter_id
),
ranked AS (
  SELECT year, fighter_id, fights_count,
         ROW_NUMBER() OVER (PARTITION BY year ORDER BY fights_count DESC) AS rn
  FROM per_year
)
SELECT r.year, 
	   r.fighter_id, 
	   pl.name, 
	   r.fights_count
FROM ranked r
LEFT JOIN fighters pl ON r.fighter_id = pl.id
WHERE r.rn <= 10
ORDER BY r.year, r.fights_count DESC;
```

## **Step 7: Data Visualisation**
Data visualization on SQL query outputs using Seaborn and Plotly: Seaborn for polished, static charts suited to documentation and reporting; Plotly for interactive, dynamic visuals suited to exploration and presentations. A complementary approach, with Seaborn ensuring reproducibility and Plotly enhancing engagement.

### *Step 7(a): Visualising Query 1*

#### Seaborn Plot
```python
# Query 1: Most Common Finishes (Seaborn)
query1 = """
SELECT method, COUNT(*) AS count
FROM fights
GROUP BY method
ORDER BY count DESC
LIMIT 20;
"""
df1 = pd.read_sql(query1, conn)

plt.figure(figsize=(10,6))
sns.barplot(df1, y = 'method', x = 'count', hue = 'method', palette = 'pastel')
plt.title("Top Finish Methods", fontsize=14)
plt.xlabel("Number of Fights")
plt.ylabel("Finish Method")
```

#### Plotly Plot
```python
# Query 2: Distribution of Fighters by Weight Class (Plotly)
# Query 1: Most Common Finishes (Plotly)
query1 = """
SELECT method, 
       COUNT(*) AS count
FROM fights
GROUP BY method
ORDER BY count DESC
LIMIT 20;
"""
df1 = pd.read_sql(query1, conn)

fig1 = px.bar(
    df1, x = 'method', y = 'count',
    title='Top Finish Methods',
    color = 'count',
    color_continuous_scale = px.colors.sequential.Viridis
)
```


### *Step 7(b): Visualising Query 2*

#### Seaborn Plot
```python
# Query 2: Distribution of Fighters by Weight Class (Seaborn)
query2 = """
SELECT COALESCE(weight_class, 'Unknown') AS weight_class,
       COUNT(*) AS n_fighters
FROM fighters
GROUP BY COALESCE(weight_class, 'Unknown')
ORDER BY n_fighters DESC;
"""
df2 = pd.read_sql(query2, conn)

plt.figure(figsize=(8,8))
colors = sns.color_palette('Set3', len(df2))
plt.pie(df2['n_fighters'], labels = df2['weight_class'], autopct = '%1.1f%%', colors = colors)
plt.title('Distribution of Fighters by Weight Class', fontsize=14)
```

#### Plotly Plot
```python
# Query 2: Distribution of Fighters by Weight Class (Plotly)
query2 = """
SELECT COALESCE(weight_class, 'Unknown') AS weight_class,
       COUNT(*) AS n_fighters
FROM fighters
GROUP BY COALESCE(weight_class, 'Unknown')
ORDER BY n_fighters DESC;
"""
df2 = pd.read_sql(query2, conn)

fig2 = px.pie(
    df2, names = 'weight_class', values = 'n_fighters',
    title = 'Distribution of Fighters by Weight Class',
    color_discrete_sequence = px.colors.qualitative.Set3
)
```

### *Step 7(c): Visualising Query 3*

#### Seaborn Plot
```python
# Query 3: Top Fighters by Win Ratio (Seaborn)
query3 = """
SELECT id, name, experience, win_ratio
FROM fighters
WHERE COALESCE(experience,0) >= 5
ORDER BY win_ratio DESC
LIMIT 20;
"""
df3 = pd.read_sql(query3, conn)

plt.figure(figsize=(10,6))
sns.barplot(df3, x = 'win_ratio', y = 'name', hue = 'name', palette = 'viridis')
plt.title('Top Fighters by Win Ratio', fontsize = 14, weight ='bold')
plt.xlabel('Win Ratio')
plt.ylabel('Fighter')
```

#### Plotly Plot
```python
# Query 3: Top Fighters by Win Ratio (Plotly)
query3 = """
SELECT id, name, experience, win_ratio
FROM fighters
WHERE COALESCE(experience,0) >= 5
ORDER BY win_ratio DESC
LIMIT 20;
"""
df3 = pd.read_sql(query3, conn)


fig3 = px.bar(
    df3.sort_values('win_ratio', ascending = True),  
    x = 'win_ratio',
    y = 'name',
    orientation = 'h',
    color = 'experience',
    color_continuous_scale = px.colors.sequential.Magma,  
    title = 'Top Fighters by Win Ratio'
)

fig3.update_layout(
    xaxis_title = 'Win Ratio',
    yaxis_title = 'Fighter',
    template = 'plotly_white',
    height=700
)

```


### *Step 7(d): Visualising Query 4*

#### Seaborn Plot
```python
# Query 4: Number of Fights per Year (Seaborn)
query4 = """
SELECT EXTRACT(YEAR FROM e.date)::INT AS year,
       COUNT(*) AS num_fights
FROM fights f
JOIN events e ON f.event_id = e.event_id
GROUP BY year
ORDER BY year;
"""
df4 = pd.read_sql(query4, conn)

plt.figure(figsize=(10,6))
sns.lineplot(df4, x = 'year', y = 'num_fights', marker = 'o')
plt.title('Number of Fights per Year', fontsize = 14, weight = 'bold')
plt.xlabel('Year')
plt.ylabel('Number of Fights')
```

#### Plotly Plot
```python
# Query 4: Number of Fights per Year (Plotly)
query4 = """
SELECT EXTRACT(YEAR FROM e.date)::INT AS year,
       COUNT(*) AS num_fights
FROM fights f
JOIN events e ON f.event_id = e.event_id
GROUP BY year
ORDER BY year;
"""
df4 = pd.read_sql(query4, conn)

fig4 = px.line(
    df4, x = 'year', y = 'num_fights',
    title = 'Fights Per Year',
    markers = True,
    color_discrete_sequence = ['#1f77b4']
)
```



### *Step 7(e): Visualising Query 5*

#### Seaborn Plot
```python
# Query 5 : Finish Method Counts + Percentage (Seaborn)
query5 = """
SELECT COALESCE(method,'Unknown') AS method,
       COUNT(*) AS cnt,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fights), 2) AS pct_of_all_fights
FROM fights
GROUP BY COALESCE(method,'Unknown')
ORDER BY cnt DESC;
"""
df5 = pd.read_sql(query5, conn)

plt.figure(figsize=(8,8))
sns.barplot(df5, x = 'pct_of_all_fights', y = 'method', hue = 'method', palette="coolwarm")
plt.title('Finishing Methods as % of All Fights', fontsize = 14, weight = 'bold')
plt.xlabel('% of All Fights')
plt.ylabel('Method')
```

#### Plotly Plot
```python
# Query 5 : Finish Method Counts + Percentage (Plotly)
query5 = """
SELECT COALESCE(method,'Unknown') AS method,
       COUNT(*) AS cnt,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fights), 2) AS pct_of_all_fights
FROM fights
GROUP BY COALESCE(method,'Unknown')
ORDER BY cnt DESC;
"""
df5 = pd.read_sql(query5, conn)

fig5 = px.bar(
    df5, x = 'method', y = 'cnt',
    title = 'Finish Methods and Percentages',
    color = 'pct_of_all_fights',
    color_continuous_scale = px.colors.sequential.Plasma
)
```



### *Step 7(f): Visualising Query 6*

#### Seaborn Plot
```python
# Query 6 : Win Ratios by Weight Class (Seaborn)
query6 = """
SELECT weight_class, COUNT(*) AS n_fighters, ROUND(AVG(win_ratio)::numeric,3) AS avg_win_ratio
FROM fighters
GROUP BY weight_class
HAVING COUNT(*) >= 5
ORDER BY avg_win_ratio DESC;
"""
df6 = pd.read_sql(query6, conn)

plt.figure(figsize=(10,6))
sns.scatterplot(df6, x = 'n_fighters', y = 'avg_win_ratio', size = 'n_fighters',
                hue = 'weight_class', palette = 'tab10', legend = False, sizes = (50, 500))
plt.title('Win Ratios by Weight Class', fontsize = 14, weight = 'bold')
plt.xlabel('Number of Fighters')
plt.ylabel('Average Win Ratio')

```

#### Plotly Plot
```python
# Query 6 : Win Ratios by Weight Class (Plotly)
query6 = """
SELECT weight_class, COUNT(*) AS n_fighters, ROUND(AVG(win_ratio)::numeric,3) AS avg_win_ratio
FROM fighters
GROUP BY weight_class
HAVING COUNT(*) >= 5
ORDER BY avg_win_ratio DESC;
"""
df6 = pd.read_sql(query6, conn)

fig6 = px.scatter(
    df6, x = 'n_fighters', y = 'avg_win_ratio', size = 'n_fighters',
    color = 'weight_class',
    title = 'Average Win Ratio by Weight Class',
    color_discrete_sequence = px.colors.qualitative.Bold
)
```

### *Step 7(g): Visualising Query 7*

#### Seaborn Plot
```python
# Query 6 : Win Ratios by Weight Class (Plotly)
query6 = """
SELECT weight_class, COUNT(*) AS n_fighters, ROUND(AVG(win_ratio)::numeric,3) AS avg_win_ratio
FROM fighters
GROUP BY weight_class
HAVING COUNT(*) >= 5
ORDER BY avg_win_ratio DESC;
"""
df6 = pd.read_sql(query6, conn)

fig6 = px.scatter(
    df6, x = 'n_fighters', y = 'avg_win_ratio', size = 'n_fighters',
    color = 'weight_class',
    title = 'Average Win Ratio by Weight Class',
    color_discrete_sequence = px.colors.qualitative.Bold
```

#### Plotly Plot
```python
# Query 7 : Fights per Year + 3-Year Moving Average (Plotly)
query7 = """
WITH per_year AS (
  SELECT DATE_TRUNC('year', e.date)::date AS year,
         COUNT(*) AS num_fights
  FROM fights f
  JOIN events e ON f.event_id = e.event_id
  GROUP BY year
)
SELECT year,
       num_fights,
       ROUND(AVG(num_fights) OVER (ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric,2) AS ma_3yr
FROM per_year
ORDER BY year;
"""
df7 = pd.read_sql(query7, conn)

fig7 = px.line(
    df7, x = 'year', y = ['num_fights', 'ma_3yr'],
    title = 'Fights Per Year with 3-Year Moving Average',
    markers = True,
    color_discrete_sequence = ['#ff7f0e', '#2ca02c']
)
```

---

## **Key Results & Insights**

1. **Dominant fight methods:** The dataset’s top fight outcome methods are summarized by a `GROUP BY method` query. The notebook visualizes method frequency and confirms the top method(s) with both Seaborn and Plotly. This reveals the distribution of fight outcomes and highlights the most common ways fights end.  

2. **Weight-class performance:** Aggregating fighters by weight class (with `HAVING COUNT(*) >= 5`) surfaces which weight classes have higher average win ratios. This identifies classes where fighters are relatively more dominant or competitive.  

3. **Experienced fighters outperform on average:** Filtering fighters with `experience >= 5` and ranking by `win_ratio` highlights top performers, useful for profiling elite fighters or scouting.  

4. **Event activity trends:** Yearly event counts computed with `DATE_TRUNC('year', date)` plus a 3‑year moving average show temporal trends in event frequency (growth, stability, or decline).  

5. **SQL ↔ pandas parity validated:** For each reporting query, a programmatic assert confirms equality between SQL aggregate results and in-memory pandas groupby results. This reduces the risk of silent aggregation or join bugs in downstream reporting.

## **Seaborn vs Plotly**

#### 1. **Seaborn (static)**

* *Purpose*: statistical data visualization built on Matplotlib, optimized for clean, publication-ready charts.
* *Pros*: compact syntax, consistent styling, wide support for statistical plots (regression, distributions, categorical comparisons), lightweight PNG outputs, reliable rendering even for medium-to-large datasets.
* *Cons*: no interactivity, limited flexibility for dynamic exploration, slower customization for highly complex visuals.

#### 2. **Plotly (interactive)**

* *Purpose*: browser-based interactive visualization, designed for exploratory analysis and dashboards.
* *Pros*: native interactivity (hover, zoom, filter), strong support for a broad range of plots (from basic scatter to 3D and geo-plots), precise rendering of large numeric ranges, self-contained HTML exports ideal for demos.
* *Cons*: heavier output size, browser-side rendering less efficient for very large datasets unless pre-aggregated, customization can be verbose.

#### 3. **General comparison**

* *Precision:* Plotly offers higher rendering precision for continuous scales and large numeric ranges; Seaborn is precise enough for standard analytic use but limited by static rendering.
* *Scale:* Seaborn handles moderate-to-large datasets efficiently in static form; Plotly can struggle with raw, very large datasets but excels once data is aggregated.
* *Variety of plots:* Plotly covers a wider spectrum (3D, maps, interactive dashboards), whereas Seaborn specializes in statistical and exploratory chart types.

#### 4. **Project context**

* *Documentation and reproducibility* → Seaborn superior: static PNGs, consistent aesthetics, integration into README and portfolio materials.
* *Live demos and exploration* → Plotly superior: interactivity, drill-down capability, engaging walkthroughs for interviews or stakeholder reviews.
* *Recommended approach* → both in tandem: Seaborn for clarity and reproducibility, Plotly for engagement and exploration. This dual approach is already implemented in the notebook and positions the project as both technically rigorous and presentation-ready.


## **Conclusion**
This repository is a concise demonstration of practical data-work skill: robust ingestion of multiple CSVs, defensible cleaning choices, purposeful feature engineering, and SQL-first aggregation validated by programmatic checks. The visual layer is intentionally dual: **Seaborn** for crisp, static documentation; **Plotly** for interactive demonstration. Together they provide both technical rigor and presentation polish.

## **Requirements**
- Python 3.8+
- Jupyter Notebook or Jupyter Lab
- Python Libraries: pandas, numpy, seaborn, matplotlib, plotly
- PostgreSQL (optional if you want to run SQL step locally)
- psycopg2 (if connecting Python to PostgreSQL)

## **Limitations & Assumptions**
- **Date quality:** date parsing uses `errors='coerce'`. The notebook documents how missing or invalid dates are handled.
- **Imputation rules:** numeric coercion can introduce `NaN` values; the notebook’s cleaning cells explicitly list fill strategies (zeros, medians, or row drop).
- **Client-side rendering limits:** Plotly is client-side; very large tables can degrade browser performance. Pre-aggregate in SQL for dashboard-ready outputs.  
- **Reproducibility:** the notebook runs top→bottom in a recommended environment; changing package versions or not following the documented cleaning steps can produce different results. Use pinned dependencies for reproducible runs.

## **Future Scope**
- **Predictive Modeling:**
Integration of a simple machine learning model (e.g., logistic regression) to predict fight outcomes using fighter statistics. Serves as a natural extension from descriptive to predictive analytics, adding interpretability and demonstrating applied ML skills.

- **Interactive Dashboard:**
Development of a Streamlit or Jupyter-based dashboard with filters by fighter, event, or weight class. Provides an accessible, interactive layer for exploring key metrics, enhancing stakeholder engagement and making the analysis more presentation-ready.

- **Automated Data Refresh:**
Implementation of a pipeline that ingests new fight, fighter, and event data as it becomes available. Ensures the analysis remains current, reproducible, and adaptable to future datasets, strengthening the project’s real-world applicability.

## **Author -** *Mayank Ahuja*
This project is part of my portfolio, demonstrating Python and SQL skills applied to a UFC dataset of events, fights and fighters, highlighting the ability to extract actionable insights from complex datasets.

![UFC](https://github.com/mayank1ahuja/da_ufc/blob/cd7636fe86d984715c6f2dcf15051eea842bc89c/project%20images/ufc.jpg)
