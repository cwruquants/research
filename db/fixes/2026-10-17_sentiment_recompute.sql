-- Applied once on 2026-10-17: recompute sentiment and readability for 1,254 old-layout
-- transcripts. The new values were computed locally, bulk-loaded into the audit table
-- sentiment_recompute_20261017 (the bulk import logged no SQL), then written with a
-- guarded UPDATE. Backups: data/private/db_fixes/sentiment_recompute_2026-10-17_*.csv

-- [2026-10-17T18:56]
-- audit table for the 2026-10-17 sentiment/readability recompute of 1,254 old-layout transcripts
CREATE TABLE public.sentiment_recompute_20261017 (
	id bigint PRIMARY KEY,
	source_file text,
	old_num_sentences integer, old_num_words integer,
	old_sentiment double precision, old_ml_score double precision, old_lm_score double precision, old_hiv4_score double precision,
	old_flesch_reading_ease double precision, old_flesch_kincaid_grade double precision, old_smog_index double precision,
	old_coleman_liau_index double precision, old_automated_readability double precision, old_dale_chall_score double precision,
	old_difficult_words integer, old_linsear_write_formula double precision, old_gunning_fog double precision, old_text_standard text,
	new_num_sentences integer, new_num_words integer,
	new_sentiment double precision, new_ml_score double precision, new_lm_score double precision, new_hiv4_score double precision,
	new_flesch_reading_ease double precision, new_flesch_kincaid_grade double precision, new_smog_index double precision,
	new_coleman_liau_index double precision, new_automated_readability double precision, new_dale_chall_score double precision,
	new_difficult_words integer, new_linsear_write_formula double precision, new_gunning_fog double precision, new_text_standard text
);

-- [2026-10-17T18:56] bulk CSV load of the recomputed values into public.sentiment_recompute_20261017 (no SQL logged)

-- [2026-10-17T19:35]
-- write recomputed sentiment/readability for the 1,254 old-layout transcripts
-- guard: all 1,254 live rows must still equal the backed-up values (fingerprint taken before the recompute)
UPDATE public.documents d
SET
	num_sentences = s.new_num_sentences,
	num_words = s.new_num_words,
	sentiment = s.new_sentiment,
	ml_score = s.new_ml_score,
	lm_score = s.new_lm_score,
	hiv4_score = s.new_hiv4_score,
	flesch_reading_ease = s.new_flesch_reading_ease,
	flesch_kincaid_grade = s.new_flesch_kincaid_grade,
	smog_index = s.new_smog_index,
	coleman_liau_index = s.new_coleman_liau_index,
	automated_readability = s.new_automated_readability,
	dale_chall_score = s.new_dale_chall_score,
	difficult_words = s.new_difficult_words,
	linsear_write_formula = s.new_linsear_write_formula,
	gunning_fog = s.new_gunning_fog,
	text_standard = s.new_text_standard
FROM public.sentiment_recompute_20261017 s
WHERE d.id = s.id
	AND (SELECT count(*) FROM public.sentiment_recompute_20261017) = 1254
	AND (
		SELECT md5(string_agg(row(x.id, x.num_sentences, x.num_words, x.sentiment, x.ml_score, x.lm_score, x.hiv4_score,
			x.flesch_reading_ease, x.flesch_kincaid_grade, x.smog_index, x.coleman_liau_index, x.automated_readability,
			x.dale_chall_score, x.difficult_words, x.linsear_write_formula, x.gunning_fog, x.text_standard)::text, '|' ORDER BY x.id))
		FROM public.documents x
		WHERE x.id IN (SELECT id FROM public.sentiment_recompute_20261017)
	) = '9bd7993367f9ac1559643ace798e4f20';
