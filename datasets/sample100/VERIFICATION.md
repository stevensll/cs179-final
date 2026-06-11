# Sample100 download verification — 2026-06-10

Verification pass per `tools/DATASET.md`, run by the download agent on titan.
Audio fetched with yt-dlp 2026.06.09 via `tools/scrape_sample100.py`
(`ytsearch1:"<artist> - <title>"`, 3 s sleep between downloads, no parallelism).
Flagged files were NOT deleted — they are listed below for manual re-sourcing.

## Summary

| Metric | Count |
|---|---|
| Tracks expected (meta/tracks.csv) | 144 |
| WAVs downloaded (audio/T###.wav) | 143 |
| Missing | 1 (T157) |
| Flagged for human review | 5 |
| Benign keyword hits (no action) | 1 (T032) |
| Duration mismatches > 5 s (ffprobe vs yt_duration_s) | 0 |

Total audio size: 5.6 GB. All 143 WAVs ffprobe-decoded cleanly; every WAV's
duration matches the logged YouTube duration within 5 s.

## Missing

| track_id | Expected | Reason |
|---|---|---|
| T157 | Geto Boys - Fuck a War | YouTube age restriction ("Sign in to confirm your age", video Ybvbnqatz1w). Retried across multiple runs; consistently blocked. Needs authenticated cookies or another source — left to human. |

## Flagged for review (re-source manually; files kept on disk)

| track_id | Expected (artist - title) | Actual yt_title | yt / wav duration | Why flagged |
|---|---|---|---|---|
| T108 | Gary Numan - Films | Gary Numan - Films (Live at OVO Arena Wembley, 2022) | 256 s / 256 s | **Live version** (2022 concert), not the 1979 studio recording. Almost certainly unusable for sample matching. |
| T109 | DJ Qbert - Eight | DJ QBERT'S MAD STUPID 8 SECOND VIDEO FOR 8.8.08!!!! | 8 s / 8 s | **8-second file** — wrong video entirely. |
| T115 | Salt-N-Pepa - I Desire | I Desiresire - Salt-N-Pepa - Topic & Salt-N-Pepa - Topic \| RaveDJ | 186 s / 186 s | **RaveDJ auto-generated mashup**, not the original recording. |
| T056 | Blowfly - Outro | Outro | 69 s / 69 s | Short file (< 90 s). May be genuine — "Outro" is a short skit-type track — but verify it is the Blowfly track (title carries no artist). |
| T146 | De La Soul - De la Orgee | De La Soul - De La Orgee (Official Audio) | 74 s / 74 s | Short file (< 90 s). Title and channel look correct; the track is genuinely ~1:14. Likely fine, flagged only by the duration rule. |

## Benign keyword hit (no action needed)

| track_id | Expected | Actual yt_title | Note |
|---|---|---|---|
| T032 | A Tribe Called Quest ft. Leaders of the New School & Kid Hood - Scenario (Remix) | A TRIBE CALLED QUEST ft KID HOOD and LEADERS OF THE NEW SCHOOL - scenario remix | "remix" keyword matched, but the expected title IS "Scenario (Remix)". Correct file. |

## Checks performed

1. **Counts**: 143 `audio/T###.wav` vs 144 track_ids in `meta/tracks.csv`;
   the only missing ID is T157 (track IDs in tracks.csv are non-contiguous —
   e.g. there is no T019; 144 is the true total).
2. **Duration sanity**: ffprobe duration of every WAV compared against the
   `yt_duration_s` column of `download_log.csv` (latest entry per track).
   No file deviates by more than 5 s. Files < 90 s or > 10 min flagged above
   (T109, T056, T146 short; none over 10 min — longest is T083 James Brown
   "Funky Drummer" full version at 556 s, which is the correct full take).
3. **Title screening**: every `yt_title` scanned (case-insensitive, word-boundary)
   for: cover, live, remix, sped, slowed, 8d, reverb, karaoke, reaction,
   tutorial, lesson, loop, sample pack, type beat. Hits: T108 (live),
   T032 (remix — benign, see above).
4. **Title overlap**: expected "artist - title" tokenized against `yt_title`
   (stopwords and uploader boilerplate like "official/audio/video/remastered/
   topic" ignored); low-overlap entries reviewed by hand. Most are
   auto-generated "Topic" uploads titled with the song name only — correct
   content. The one real catch was T115 (RaveDJ mashup). Borderline but
   accepted: T078 ("UK New Entry 1969 (14) Sam & Dave - Soul Sister, Brown
   Sugar", 149 s — looks like a chart-countdown upload of the studio
   recording; spot-listen if T078's pairs misbehave).

## Notes

- `download_log.csv` contains repeated `download_failed` rows for T157 — one
  per resume run; that is expected behavior of the idempotent scraper.
- No throttling, HTTP 403, or bot-detection encountered; all 143 downloads
  succeeded on the first attempt at 3 s spacing.
- Affected eval pairs: any row of `eval_pairs.csv` referencing T056, T108,
  T109, T115, T146 (suspect audio) or T157 (missing) should be excluded or
  hand-fixed before scoring runs.
