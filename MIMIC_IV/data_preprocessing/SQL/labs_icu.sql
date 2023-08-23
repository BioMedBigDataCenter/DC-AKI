SET search_path = public, mimiciv_derived, mimiciv_hosp, mimiciv_icu;

-- This query pivots lab values taken in the all time of a patient's stay
-- Have already confirmed that the unit of measurement is always the same: null or the correct unit

drop table if exists labs_all_icu_mean_0721;
create table labs_all_icu_mean_0721 as
(
	SELECT
	  pvt.subject_id, pvt.hadm_id, pvt.stay_id, pvt.charttime

-- 	  , min(CASE WHEN label = 'ANION GAP' THEN valuenum ELSE NULL END) AS aniongap_min
-- 	  , max(CASE WHEN label = 'ANION GAP' THEN valuenum ELSE NULL END) AS aniongap_max
	  , avg(CASE WHEN label = 'ANION GAP' THEN valuenum ELSE NULL END) AS aniongap_mean
-- 	  , min(CASE WHEN label = 'ALBUMIN' THEN valuenum ELSE NULL END) AS albumin_min
-- 	  , max(CASE WHEN label = 'ALBUMIN' THEN valuenum ELSE NULL END) AS albumin_max
	  , avg(CASE WHEN label = 'ALBUMIN' THEN valuenum ELSE NULL END) AS albumin_mean
-- 	  , min(CASE WHEN label = 'BANDS' THEN valuenum ELSE NULL END) AS bands_min
-- 	  , max(CASE WHEN label = 'BANDS' THEN valuenum ELSE NULL END) AS bands_max
	  , avg(CASE WHEN label = 'BANDS' THEN valuenum ELSE NULL END) AS bands_mean
-- 	  , min(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE NULL END) AS bicarbonate_min
-- 	  , max(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE NULL END) AS bicarbonate_max
	  , avg(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE NULL END) AS bicarbonate_mean
-- 	  , min(CASE WHEN label = 'BILIRUBIN' THEN valuenum ELSE NULL END) AS bilirubin_min
-- 	  , max(CASE WHEN label = 'BILIRUBIN' THEN valuenum ELSE NULL END) AS bilirubin_max
	  , avg(CASE WHEN label = 'BILIRUBIN' THEN valuenum ELSE NULL END) AS bilirubin_mean
-- 	  , min(CASE WHEN label = 'CREATININE' THEN valuenum ELSE NULL END) AS creatinine_min
-- 	  , max(CASE WHEN label = 'CREATININE' THEN valuenum ELSE NULL END) AS creatinine_max
	  , avg(CASE WHEN label = 'CREATININE' THEN valuenum ELSE NULL END) AS creatinine_mean
-- 	  , min(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE NULL END) AS chloride_min
-- 	  , max(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE NULL END) AS chloride_max
	  , avg(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE NULL END) AS chloride_mean
-- 	  , min(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE NULL END) AS glucose_min
-- 	  , max(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE NULL END) AS glucose_max
	  , avg(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE NULL END) AS glucose_mean
-- 	  , min(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE NULL END) AS hematocrit_min
-- 	  , max(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE NULL END) AS hematocrit_max
	  , avg(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE NULL END) AS hematocrit_mean
-- 	  , min(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE NULL END) AS hemoglobin_min
-- 	  , max(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE NULL END) AS hemoglobin_max
	  , avg(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE NULL END) AS hemoglobin_mean
-- 	  , min(CASE WHEN label = 'LACTATE' THEN valuenum ELSE NULL END) AS lactate_min
-- 	  , max(CASE WHEN label = 'LACTATE' THEN valuenum ELSE NULL END) AS lactate_max
	  , avg(CASE WHEN label = 'LACTATE' THEN valuenum ELSE NULL END) AS lactate_mean
-- 	  , min(CASE WHEN label = 'PLATELET' THEN valuenum ELSE NULL END) AS platelet_min
-- 	  , max(CASE WHEN label = 'PLATELET' THEN valuenum ELSE NULL END) AS platelet_max
	  , avg(CASE WHEN label = 'PLATELET' THEN valuenum ELSE NULL END) AS platelet_mean
-- 	  , min(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE NULL END) AS potassium_min
-- 	  , max(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE NULL END) AS potassium_max
	  , avg(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE NULL END) AS potassium_mean
-- 	  , min(CASE WHEN label = 'PTT' THEN valuenum ELSE NULL END) AS ptt_min
-- 	  , max(CASE WHEN label = 'PTT' THEN valuenum ELSE NULL END) AS ptt_max
	  , avg(CASE WHEN label = 'PTT' THEN valuenum ELSE NULL END) AS ptt_mean
-- 	  , min(CASE WHEN label = 'INR' THEN valuenum ELSE NULL END) AS inr_min
-- 	  , max(CASE WHEN label = 'INR' THEN valuenum ELSE NULL END) AS inr_max
	  , avg(CASE WHEN label = 'INR' THEN valuenum ELSE NULL END) AS inr_mean
-- 	  , min(CASE WHEN label = 'PT' THEN valuenum ELSE NULL END) AS pt_min
-- 	  , max(CASE WHEN label = 'PT' THEN valuenum ELSE NULL END) AS pt_max
	  , avg(CASE WHEN label = 'PT' THEN valuenum ELSE NULL END) AS pt_mean
-- 	  , min(CASE WHEN label = 'SODIUM' THEN valuenum ELSE NULL END) AS sodium_min
-- 	  , max(CASE WHEN label = 'SODIUM' THEN valuenum ELSE NULL END) AS sodium_max
	  , avg(CASE WHEN label = 'SODIUM' THEN valuenum ELSE NULL END) AS sodium_mean
-- 	  , min(CASE WHEN label = 'BUN' THEN valuenum ELSE NULL END) AS bun_min
-- 	  , max(CASE WHEN label = 'BUN' THEN valuenum ELSE NULL END) AS bun_max
	  , avg(CASE WHEN label = 'BUN' THEN valuenum ELSE NULL END) AS bun_mean
-- 	  , min(CASE WHEN label = 'WBC' THEN valuenum ELSE NULL END) AS wbc_min
-- 	  , max(CASE WHEN label = 'WBC' THEN valuenum ELSE NULL END) AS wbc_max
	  , avg(CASE WHEN label = 'WBC' THEN valuenum ELSE NULL END) AS wbc_mean

	FROM
	( -- begin query that extracts the data
	  SELECT ie.subject_id, ie.hadm_id, ie.stay_id, ie.intime, ie.outtime, le.charttime
	  -- here we assign labels to ITEMIDs
	  -- this also fuses together multiple ITEMIDs containing the same data
	  , CASE
			WHEN itemid = 50868 THEN 'ANION GAP'
			WHEN itemid = 50862 THEN 'ALBUMIN'
			WHEN itemid = 51144 THEN 'BANDS'
			WHEN itemid = 50882 THEN 'BICARBONATE'
			WHEN itemid = 50885 THEN 'BILIRUBIN'
			WHEN itemid = 50912 THEN 'CREATININE'
			WHEN itemid = 50806 THEN 'CHLORIDE'
			WHEN itemid = 50902 THEN 'CHLORIDE'
			WHEN itemid = 50809 THEN 'GLUCOSE'
			WHEN itemid = 50931 THEN 'GLUCOSE'
			WHEN itemid = 50810 THEN 'HEMATOCRIT'
			WHEN itemid = 51221 THEN 'HEMATOCRIT'
			WHEN itemid = 50811 THEN 'HEMOGLOBIN'
			WHEN itemid = 51222 THEN 'HEMOGLOBIN'
			WHEN itemid = 50813 THEN 'LACTATE'
			WHEN itemid = 51265 THEN 'PLATELET'
			WHEN itemid = 50822 THEN 'POTASSIUM'
			WHEN itemid = 50971 THEN 'POTASSIUM'
			WHEN itemid = 51275 THEN 'PTT'
			WHEN itemid = 51237 THEN 'INR'
			WHEN itemid = 51274 THEN 'PT'
			WHEN itemid = 50824 THEN 'SODIUM'
			WHEN itemid = 50983 THEN 'SODIUM'
			WHEN itemid = 51006 THEN 'BUN'
			WHEN itemid = 51300 THEN 'WBC'
			WHEN itemid = 51301 THEN 'WBC'
		  ELSE null
		END as label
	  , -- add in some sanity checks on the values
	  -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
		CASE
		  WHEN itemid = 50862 and valuenum >    10 THEN null -- g/dL 'ALBUMIN'
		  WHEN itemid = 50868 and valuenum > 10000 THEN null -- mEq/L 'ANION GAP'
		  WHEN itemid = 51144 and valuenum <     0 THEN null -- immature band forms, %
		  WHEN itemid = 51144 and valuenum >   100 THEN null -- immature band forms, %
		  WHEN itemid = 50882 and valuenum > 10000 THEN null -- mEq/L 'BICARBONATE'
		  WHEN itemid = 50885 and valuenum >   150 THEN null -- mg/dL 'BILIRUBIN'
		  WHEN itemid = 50806 and valuenum > 10000 THEN null -- mEq/L 'CHLORIDE'
		  WHEN itemid = 50902 and valuenum > 10000 THEN null -- mEq/L 'CHLORIDE'
		  WHEN itemid = 50912 and valuenum >   150 THEN null -- mg/dL 'CREATININE'
		  WHEN itemid = 50809 and valuenum > 10000 THEN null -- mg/dL 'GLUCOSE'
		  WHEN itemid = 50931 and valuenum > 10000 THEN null -- mg/dL 'GLUCOSE'
		  WHEN itemid = 50810 and valuenum >   100 THEN null -- % 'HEMATOCRIT'
		  WHEN itemid = 51221 and valuenum >   100 THEN null -- % 'HEMATOCRIT'
		  WHEN itemid = 50811 and valuenum >    50 THEN null -- g/dL 'HEMOGLOBIN'
		  WHEN itemid = 51222 and valuenum >    50 THEN null -- g/dL 'HEMOGLOBIN'
		  WHEN itemid = 50813 and valuenum >    50 THEN null -- mmol/L 'LACTATE'
		  WHEN itemid = 51265 and valuenum > 10000 THEN null -- K/uL 'PLATELET'
		  WHEN itemid = 50822 and valuenum >    30 THEN null -- mEq/L 'POTASSIUM'
		  WHEN itemid = 50971 and valuenum >    30 THEN null -- mEq/L 'POTASSIUM'
		  WHEN itemid = 51275 and valuenum >   150 THEN null -- sec 'PTT'
		  WHEN itemid = 51237 and valuenum >    50 THEN null -- 'INR'
		  WHEN itemid = 51274 and valuenum >   150 THEN null -- sec 'PT'
		  WHEN itemid = 50824 and valuenum >   200 THEN null -- mEq/L == mmol/L 'SODIUM'
		  WHEN itemid = 50983 and valuenum >   200 THEN null -- mEq/L == mmol/L 'SODIUM'
		  WHEN itemid = 51006 and valuenum >   300 THEN null -- 'BUN'
		  WHEN itemid = 51300 and valuenum >  1000 THEN null -- 'WBC'
		  WHEN itemid = 51301 and valuenum >  1000 THEN null -- 'WBC'
		ELSE le.valuenum
		END as valuenum

	  FROM icustays ie

	  LEFT JOIN labevents le
		ON ie.hadm_id = le.hadm_id
	    AND le.charttime BETWEEN ie.intime and ie.outtime
		AND le.ITEMID in
		(
		  -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
		  50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
		  50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
		  51144, -- BANDS - hematology
		  50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
		  50885, -- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
		  50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
		  50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
		  50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
		  50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
		  50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
		  51221, -- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
		  50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
		  51222, -- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
		  50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
		  50813, -- LACTATE | BLOOD GAS | BLOOD | 187124
		  51265, -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
		  50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
		  50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
		  51275, -- PTT | HEMATOLOGY | BLOOD | 474937
		  51237, -- INR(PT) | HEMATOLOGY | BLOOD | 471183
		  51274, -- PT | HEMATOLOGY | BLOOD | 469090
		  50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
		  50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
		  51006, -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
		  51301, -- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
		  51300  -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
		)
		AND valuenum IS NOT null AND valuenum > 0 -- lab values cannot be 0 and cannot be negative
	) pvt
	where pvt.stay_id in (select distinct stay_id from sample_inclusion_0721)
	GROUP BY pvt.subject_id, pvt.hadm_id, pvt.stay_id, pvt.charttime
	ORDER BY pvt.subject_id, pvt.hadm_id, pvt.stay_id, pvt.charttime
);
