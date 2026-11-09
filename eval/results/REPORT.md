# Ablation report

Hit = within ±5 s of the frame-verified event time. Recall is relative to the pooled truth (see eval/README.md). Cost columns are from the run that produced the predictions (cached slices cost 0).

## deku_tree_death

first frame the Deku Tree's line '...my time is finally coming to an end' is OCR-readable (360p)

truth: 23 present, 0 absent, 39 unknown

| config | precision | recall | hits | predictions | present | err_median_s | err_p90_s | slices | slice_mb | wall_s | http_429 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transcript | 0.0 | 0.0 | 0 | 56 | 23 | 426.9 | 2602.0 | 0 | 0.0 | 4.2 | 0 |
| cues | 0.0 | 0.0 | 0 | 56 | 23 | 426.9 | 2602.0 | 0 | 0.0 | 4.0 | 0 |
| screen | 0.667 | 0.522 | 12 | 22 | 23 | 2.9 | 10.5 | 0 | 0.0 | 0.1 | 0 |
| screen+refine | 1.0 | 0.783 | 18 | 22 | 23 | 0.0 | 0.0 | 0 | 0.0 | 14.0 | 0 |
| screen+context+refine | 1.0 | 1.0 | 23 | 23 | 23 | 0.0 | 0.0 | 0 | 0.0 | 30.6 | 0 |

## gohma

first frame the 'Parasitic Armored Arachnid / GOHMA' boss name card is OCR-readable

truth: 16 present, 0 absent, 46 unknown

| config | precision | recall | hits | predictions | present | err_median_s | err_p90_s | slices | slice_mb | wall_s | http_429 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transcript | 0.0 | 0.0 | 0 | 56 | 16 | 989.9 | 5703.5 | 0 | 0.0 | 4.2 | 0 |
| cues | 0.0 | 0.0 | 0 | 56 | 16 | 989.9 | 5703.5 | 0 | 0.0 | 4.0 | 0 |
| screen | 0.571 | 0.5 | 8 | 14 | 16 | 4.6 | 8.9 | 0 | 0.0 | 0.1 | 0 |
| screen+refine | 1.0 | 0.875 | 14 | 14 | 16 | 0.0 | 0.0 | 0 | 0.0 | 3.5 | 0 |
| screen+context+refine | 1.0 | 0.875 | 14 | 14 | 16 | 0.0 | 0.0 | 0 | 0.0 | 7.0 | 0 |

## hyrule_field

first frame the 'HYRULE FIELD' area title card is visible on first entry from Kokiri Forest

truth: 36 present, 0 absent, 26 unknown

| config | precision | recall | hits | predictions | present | err_median_s | err_p90_s | slices | slice_mb | wall_s | http_429 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transcript | 0.152 | 0.139 | 5 | 47 | 36 | 46.0 | 7350.6 | 0 | 0.0 | 4.5 | 0 |
| cues | 0.152 | 0.139 | 5 | 47 | 36 | 37.9 | 4224.2 | 0 | 0.0 | 4.3 | 0 |
| screen | 0.579 | 0.306 | 11 | 23 | 36 | 4.9 | 10407.0 | 0 | 0.0 | 0.1 | 0 |
| screen+refine | 0.842 | 0.444 | 16 | 23 | 36 | 0.0 | 10418.0 | 0 | 0.0 | 6.3 | 0 |
| screen+context+refine | 1.0 | 0.861 | 31 | 31 | 36 | 0.0 | 0.0 | 0 | 0.0 | 18.0 | 0 |
| all | 1.0 | 0.972 | 35 | 35 | 36 | 0.0 | 1.0 | 0 | 0.0 | 35.3 | 0 |

## zelda_meeting

first frame Zelda's 'ZELDA / Princess of Hyrule' character name card is OCR-readable (courtyard meeting)

truth: 5 present, 0 absent, 57 unknown

| config | precision | recall | hits | predictions | present | err_median_s | err_p90_s | slices | slice_mb | wall_s | http_429 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transcript | 0.0 | 0.0 | 0 | 56 | 5 | 501.5 | 27550.6 | 0 | 0.0 | 4.0 | 0 |
| cues | 0.0 | 0.0 | 0 | 57 | 5 | 501.5 | 27550.6 | 0 | 0.0 | 3.7 | 0 |
| screen | 0.333 | 0.4 | 2 | 6 | 5 | 7.0 | 12.8 | 0 | 0.0 | 0.1 | 0 |
| screen+refine | 0.833 | 1.0 | 5 | 6 | 5 | 0.0 | 0.0 | 1 | 0.3 | 20.3 | 0 |
| screen+context+refine | 0.833 | 1.0 | 5 | 6 | 5 | 0.0 | 0.0 | 0 | 0.0 | 7.2 | 0 |
