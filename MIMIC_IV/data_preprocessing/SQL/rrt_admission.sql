SET search_path = public, mimiciv_derived, mimiciv_hosp, mimiciv_icu;
-- determines if patients received any dialysis during their admission

drop table if exists rrt_all_admission_0721;
create table rrt_all_admission_0721 as
(
	with tmp as
	(
		with cv as
		(
		  select ie.hadm_id, ie.stay_id
			, ce.charttime as starttime
			, ce.charttime as endtime
			, max(
				case
				  when ce.itemid in (152,148,149,146,147,151,150) and value is not null then 1
				  when ce.itemid in (229,235,241,247,253,259,265,271) and value = 'Dialysis Line' then 1
				  when ce.itemid = 582 and value in ('CAVH Start','CAVH D/C','CVVHD Start','CVVHD D/C','Hemodialysis st','Hemodialysis end') then 1
				else 0 end
				) as RRT
		  FROM sample_inclusion_0721 ie
		  inner join chartevents ce
			on ie.hadm_id = ce.hadm_id
			and ce.itemid in
			(
			   152 -- "Dialysis Type";61449
			  ,148 -- "Dialysis Access Site";60335
			  ,149 -- "Dialysis Access Type";60030
			  ,146 -- "Dialysate Flow ml/hr";57445
			  ,147 -- "Dialysate Infusing";56605
			  ,151 -- "Dialysis Site Appear";37345
			  ,150 -- "Dialysis Machine";27472
			  ,229 -- INV Line#1 [Type]
			  ,235 -- INV Line#2 [Type]
			  ,241 -- INV Line#3 [Type]
			  ,247 -- INV Line#4 [Type]
			  ,253 -- INV Line#5 [Type]
			  ,259 -- INV Line#6 [Type]
			  ,265 -- INV Line#7 [Type]
			  ,271 -- INV Line#8 [Type]
			  ,582 -- Procedures
			)
			and ce.value is not null
		--     and ce.charttime between ie.intime and ie.outtime
		--   where ie.race = 'carevue'
		  group by ie.hadm_id, ie.stay_id, starttime, endtime
		)
		, mv_ce as
		(
		  select ie.hadm_id, ie.stay_id
			, ce.charttime as starttime
			, ce.charttime as endtime
			, 1 as RRT
		  FROM sample_inclusion_0721 ie
		  inner join chartevents ce
			on ie.hadm_id = ce.hadm_id
		--     and ce.charttime between ie.intime and ie.outtime
			and itemid in
			(
			  -- Checkboxes
				226118 -- | Dialysis Catheter placed in outside facility      | Access Lines - Invasive | chartevents        | Checkbox
			  , 227357 -- | Dialysis Catheter Dressing Occlusive              | Access Lines - Invasive | chartevents        | Checkbox
			  , 225725 -- | Dialysis Catheter Tip Cultured                    | Access Lines - Invasive | chartevents        | Checkbox
			  -- Numeric values
			  , 226499 -- | Hemodialysis Output                               | Dialysis                | chartevents        | Numeric
			  , 224154 -- | Dialysate Rate                                    | Dialysis                | chartevents        | Numeric
			  , 225810 -- | Dwell Time (Peritoneal Dialysis)                  | Dialysis                | chartevents        | Numeric
			  , 227639 -- | Medication Added Amount  #2 (Peritoneal Dialysis) | Dialysis                | chartevents        | Numeric
			  , 225183 -- | Current Goal                     | Dialysis | chartevents        | Numeric
			  , 227438 -- | Volume not removed               | Dialysis | chartevents        | Numeric
			  , 224191 -- | Hourly Patient Fluid Removal     | Dialysis | chartevents        | Numeric
			  , 225806 -- | Volume In (PD)                   | Dialysis | chartevents        | Numeric
			  , 225807 -- | Volume Out (PD)                  | Dialysis | chartevents        | Numeric
			  , 228004 -- | Citrate (ACD-A)                  | Dialysis | chartevents        | Numeric
			  , 228005 -- | PBP (Prefilter) Replacement Rate | Dialysis | chartevents        | Numeric
			  , 228006 -- | Post Filter Replacement Rate     | Dialysis | chartevents        | Numeric
			  , 224144 -- | Blood Flow (ml/min)              | Dialysis | chartevents        | Numeric
			  , 224145 -- | Heparin Dose (per hour)          | Dialysis | chartevents        | Numeric
			  , 224149 -- | Access Pressure                  | Dialysis | chartevents        | Numeric
			  , 224150 -- | Filter Pressure                  | Dialysis | chartevents        | Numeric
			  , 224151 -- | Effluent Pressure                | Dialysis | chartevents        | Numeric
			  , 224152 -- | Return Pressure                  | Dialysis | chartevents        | Numeric
			  , 224153 -- | Replacement Rate                 | Dialysis | chartevents        | Numeric
			  , 224404 -- | ART Lumen Volume                 | Dialysis | chartevents        | Numeric
			  , 224406 -- | VEN Lumen Volume                 | Dialysis | chartevents        | Numeric
			  , 226457 -- | Ultrafiltrate Output             | Dialysis | chartevents        | Numeric
			)
			and valuenum > 0 -- also ensures it's not null
		  group by ie.hadm_id, ie.stay_id, starttime, endtime
		)
		, mv_ie as
		(
		  select ie.hadm_id, ie.stay_id, tt.starttime, tt.endtime
			, 1 as RRT
		  FROM sample_inclusion_0721 ie
		  inner join inputevents tt
			on ie.hadm_id = tt.hadm_id
		--     and tt.starttime between ie.intime and ie.outtime
			and itemid in
			(
				227536 --	KCl (CRRT)	Medications	inputevents	Solution
			  , 227525 --	Calcium Gluconate (CRRT)	Medications	inputevents	Solution
			)
			and amount > 0 -- also ensures it's not null
		  group by ie.hadm_id, ie.stay_id, tt.starttime, tt.endtime
		)
		, mv_de as
		(
		  select ie.hadm_id, ie.stay_id
			, tt.charttime as starttime
			, tt.charttime as endtime
			, 1 as RRT
		  FROM sample_inclusion_0721 ie
		  inner join datetimeevents tt
			on ie.hadm_id = tt.hadm_id
		--     and tt.charttime between ie.intime and ie.outtime
			and itemid in
			(
			  -- TODO: unsure how to handle "Last dialysis"
			  --  225128 -- | Last dialysis                                     | Adm History/FHPA        | datetimeevents     | Date time
				225318 -- | Dialysis Catheter Cap Change                      | Access Lines - Invasive | datetimeevents     | Date time
			  , 225319 -- | Dialysis Catheter Change over Wire Date           | Access Lines - Invasive | datetimeevents     | Date time
			  , 225321 -- | Dialysis Catheter Dressing Change                 | Access Lines - Invasive | datetimeevents     | Date time
			  , 225322 -- | Dialysis Catheter Insertion Date                  | Access Lines - Invasive | datetimeevents     | Date time
			  , 225324 -- | Dialysis CatheterTubing Change                    | Access Lines - Invasive | datetimeevents     | Date time
			)
		  group by ie.hadm_id, ie.stay_id, starttime, endtime
		)
		, mv_pe as
		(
			select ie.hadm_id, ie.stay_id, tt.starttime, tt.endtime
			  , 1 as RRT
			FROM sample_inclusion_0721 ie
			inner join procedureevents tt
			  on ie.hadm_id = tt.hadm_id
		--       and tt.starttime between ie.intime and ie.outtime
			  and itemid in
			  (
				  225441 -- | Hemodialysis                                      | 4-Procedures            | procedureevents | Process
				, 225802 -- | Dialysis - CRRT                                   | Dialysis                | procedureevents | Process
				, 225803 -- | Dialysis - CVVHD                                  | Dialysis                | procedureevents | Process
				, 225805 -- | Peritoneal Dialysis                               | Dialysis                | procedureevents | Process
				, 224270 -- | Dialysis Catheter                                 | Access Lines - Invasive | procedureevents | Process
				, 225809 -- | Dialysis - CVVHDF                                 | Dialysis                | procedureevents | Process
				, 225955 -- | Dialysis - SCUF                                   | Dialysis                | procedureevents | Process
				, 225436 -- | CRRT Filter Change               | Dialysis | procedureevents | Process
			  )
			group by ie.hadm_id, ie.stay_id, tt.starttime, tt.endtime
		)
		select cv.* from cv where rrt = 1
		union all
		select mv_ce.* from mv_ce where rrt = 1
		union all
		select mv_ie.* from mv_ie where rrt = 1
		union all
		select mv_de.* from mv_de where rrt = 1
		union all
		select mv_pe.* from mv_pe where rrt = 1
		order by hadm_id, stay_id, starttime, endtime
	)
	select hadm_id, stay_id, starttime, endtime
		, max(rrt) as rrt
	from tmp
	group by hadm_id, stay_id, starttime, endtime
);

select * from sample_inclusion_0721
