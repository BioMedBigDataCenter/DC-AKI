SET search_path = public, mimiciv_derived, mimiciv_hosp, mimiciv_icu;

drop table if exists icustay_detail_0721;
create table icustay_detail_0721 as
(
	select
		ie.subject_id, ie.hadm_id, ie.stay_id,
		ie.gender, ie.admittime, ie.dischtime, ie.admission_age, ie.race, ad.admission_type
	from icustay_detail ie
	left join admissions ad
		on ie.hadm_id = ad.hadm_id
	where stay_id in (select distinct stay_id from sample_inclusion_0721)
);