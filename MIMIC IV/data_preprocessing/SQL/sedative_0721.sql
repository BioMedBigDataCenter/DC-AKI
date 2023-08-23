SET search_path = public, mimiciv_derived, mimiciv_hosp, mimiciv_icu;
DROP table IF EXISTS sedative;
CREATE table sedative as
SELECT stay_id, linkorderid,starttime,endtime, amount AS sedative_amount
FROM inputevents
where itemid in(221668,221744,225972,225942,222168)and statusdescription != 'Rewritten' -- only valid orders;