-- Step 6: Exploratory Data Analysis(EDA)

-- Creating Table Fighters
CREATE TABLE fighters (
			id VARCHAR(25),
			name VARCHAR(50),
			nick_name VARCHAR(50),
			wins INT,	
			losses INT,	
			draws INT,	
			height FLOAT,	
			weight FLOAT,	
			reach FLOAT,
			stance VARCHAR(25),	
			dob	DATE,
			splm FLOAT,	
			str_acc	INT,
			sapm FLOAT,
			str_def	INT,
			td_avg FLOAT,	
			td_avg_acc INT,	
			td_def INT,	
			sub_avg	FLOAT,
			weight_class VARCHAR(25),	
			age	FLOAT,
			win_ratio FLOAT(2),	
			experience INT
);
-- Verifying Imported Data
SELECT * FROM fighters;


-- Creating Table Fights
CREATE TABLE fights (
			event_name VARCHAR(125),
			event_id VARCHAR(50),
			fight_id VARCHAR(50),	
			r_name VARCHAR(50),		
			r_id VARCHAR(50),		
			b_name VARCHAR(50),		
			b_id VARCHAR(50),		
			division VARCHAR(100),		
			title_fight	INT,
			method VARCHAR(50),		
			finish_round INT,	
			match_time_sec INT,	
			total_rounds INT,	
			referee VARCHAR(50)
);
-- Verifying Imported Data
SELECT * FROM fights;

ALTER TABLE fights
RENAME COLUMN division TO weight_class;


-- Creating Table Events
CREATE TABLE events (
			event_id VARCHAR(50),
			fight_id VARCHAR(50),
			date DATE,
			location VARCHAR(200),
			winner VARCHAR(50),
			winner_id VARCHAR(50),
			event_year INT,
			event_month INT
);			
-- Verifying Imported Data
SELECT * FROM events;


-- QUERIES
-- Question 1 : What are the most common finish methods and how many fights ended by each method?
SELECT method, 
	   COUNT(*) AS count
FROM fights
GROUP BY method
ORDER BY count DESC
LIMIT 20;

-- Question 2 : What is the distribution of fighters by weight class?
SELECT COALESCE(weight_class, 'Unknown') AS weight_class,
       COUNT(*) AS n_fighters
FROM fighters
GROUP BY COALESCE(weight_class, 'Unknown')
ORDER BY n_fighters DESC;

-- Question 3 : Who are the top fighters by win_ratio?
SELECT id, 
	   name, 
	   experience,
	   win_ratio
FROM fighters
WHERE COALESCE(experience,0) >= 5
ORDER BY win_ratio DESC
LIMIT 20;

-- Question 4 : What are the number of fights per year?
SELECT EXTRACT(YEAR FROM e.date)::INT AS year,
       COUNT(*) AS num_fights
FROM fights f
JOIN events e ON f.event_id = e.event_id
GROUP BY year
ORDER BY year;


-- Question 5 : What are the total counts of each finishing method and what percentage are they of the total?
SELECT COALESCE(method,'Unknown') AS method,
       COUNT(*) AS cnt,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fights), 2) AS pct_of_all_fights
FROM fights
GROUP BY COALESCE(method,'Unknown')
ORDER BY cnt DESC;

-- Question 6 : Define the win ratios by weight class.
SELECT weight_class,
	   COUNT(*) AS n_fighters,
	   ROUND(AVG(win_ratio)::numeric, 3) AS avg_win_ratio
FROM fighters
GROUP BY weight_class
HAVING COUNT(*) >= 5
ORDER BY avg_win_ratio DESC;


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


















