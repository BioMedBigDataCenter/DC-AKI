SET search_path = public, mimiciv_derived, mimiciv_hosp, mimiciv_icu;
-- -- set search_path to public, mimiciv;
-- -- inclusion criteria:
-- -- 	1. age: 18-89
-- -- 	2. length of icu stay >= 48h
-- --   3. not have AKI before ICU

drop table if exists aki_before_icu_0721;
create table aki_before_icu_0721 as
select
	ie.stay_id, ie.icu_intime, ie.icu_outtime, ie.first_icu_stay, ks.charttime, ks.aki_stage
from icustay_detail ie
left join kdigo_stages ks
	on ie.stay_id = ks.stay_id
	and ks.charttime < ie.icu_intime
where aki_stage != 0;


drop table if exists sample_inclusion_0721;
create table sample_inclusion_0721 as
(
	with tmp as
	(
		select
			dtl.subject_id, dtl.hadm_id, dtl.stay_id, dtl.icu_intime, dtl.icu_outtime
		from icustay_detail dtl
			where dtl.admission_age between 18 and 89
			and dtl.los_icu >= 2
	)
	select tmp.*
	from tmp
	left join icustays ie
		on tmp.stay_id = ie.stay_id
);
delete from sample_inclusion_0721 
	where stay_id in (select distinct stay_id from aki_before_icu_0721)
		or icu_intime is null
		or icu_outtime is null;

-- KDIGO_stages
drop table if exists kdigo_stages_0721;
create table kdigo_stages_0721 as 
(
	select
		smp.*, stg.charttime, stg.creat,
		stg.uo_rt_6hr, stg.uo_rt_12hr, stg.uo_rt_24hr, stg.aki_stage
	from sample_inclusion_0721 smp
	left join kdigo_stages stg
		on smp.stay_id = stg.stay_id
		and stg.charttime is not null
		and stg.charttime between smp.icu_intime and smp.icu_outtime
);
delete from sample_inclusion_0721
	where stay_id in (select distinct stay_id from kdigo_stages_0721 where charttime is null or aki_stage is null);
delete from kdigo_stages_0721 where charttime is null or aki_stage is null;




