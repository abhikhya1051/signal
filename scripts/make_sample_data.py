"""
Generates sample_data/stride_reviews.csv: 150 synthetic app reviews for
"Stride", a fictional running & fitness app.

Five problems/themes are deliberately planted in the data with known
frequencies, so you can check whether Signal finds them (see README).

Run:  python scripts/make_sample_data.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(7)

# theme -> (how many reviews, rating range, phrasings)
THEMES = {
    "Watch sync failures": (34, (1, 2), [
        "My runs from my Garmin just don't show up anymore. I have to re-pair every few days.",
        "Apple Watch sync is broken. Did a 10k this morning and it's nowhere in the app.",
        "Workouts recorded on my watch take hours to appear, if they appear at all.",
        "Sync with my Fitbit stopped working after the last update.",
        "Lost three weeks of training history because the watch sync silently failed.",
        "The watch app says 'synced' but the phone shows nothing. Very frustrating.",
        "Heart rate data from my watch comes through but the route never does.",
        "I bought a new watch specifically for Stride and it can't stay connected.",
        "Every time I open the app I have to hit sync three times before my run shows.",
        "Duplicated runs! Some workouts sync twice, others not at all.",
        "My streak got reset because yesterday's run never synced from my watch.",
        "Sync with Garmin Connect has been flaky for a month now.",
    ]),
    "Subscription & billing": (26, (1, 3), [
        "I was charged for a full year when I thought I was on a monthly trial.",
        "Way too expensive for what you get. The price went up again without notice.",
        "Cancelling the subscription is buried five screens deep. Feels intentional.",
        "Free trial converted to paid with no reminder email. Not cool.",
        "Most features moved behind the paywall. The free version is basically useless now.",
        "Got double charged this month and support hasn't replied in a week.",
        "Love the app but $14.99 a month is hard to justify next to the competition.",
        "Please offer a cheaper plan without the meal-planning stuff I never use.",
        "The 'Pro' upsell popup appears after every single run. So pushy.",
        "Asked for a refund for an accidental renewal, got a canned response.",
    ]),
    "Crashes & slowness": (22, (1, 2), [
        "App crashes the moment I tap 'Start run'.",
        "Since version 6.2 the app freezes on the summary screen after every workout.",
        "It takes almost 20 seconds to open. It used to be instant.",
        "Crashed mid-marathon training run and I lost the whole thing.",
        "The map screen stutters constantly on my phone, battery drains fast too.",
        "Keeps crashing on Android 15, please fix.",
        "Opening the history tab makes the whole app hang.",
        "Laggy and slow after the redesign, everything takes extra taps and loads.",
    ]),
    "GPS accuracy": (16, (2, 3), [
        "GPS distance is off by almost half a mile on a 5k loop I know well.",
        "My pace jumps around wildly. It said I ran a 3-minute mile, I promise I didn't.",
        "The route map cuts corners and shows me running through buildings.",
        "Distance is always shorter than my watch, which makes my pace look worse.",
        "GPS takes forever to lock on before a run starts.",
        "Treadmill vs outdoor detection is wrong, it logged my park run as indoor.",
    ]),
    "Love the coaching plans": (30, (4, 5), [
        "The 10k training plan got me from couch to my first race. Amazing.",
        "Guided audio runs are the best feature, the coach voice keeps me going.",
        "Adaptive plan adjusted when I got sick, which was really thoughtful.",
        "I finally hit a sub-25 5k thanks to the interval workouts.",
        "The coaching tips after each run are genuinely useful, not generic.",
        "Love how the plan builds week by week, I always know what to do today.",
        "Recovery day suggestions saved me from overtraining.",
        "The half marathon plan is excellent, better than paid coaches I've tried.",
        "Clean design and the training plans are spot on.",
        "Beginner plan is very encouraging, no judgment, just progress.",
    ]),
    "Feature requests": (14, (3, 4), [
        "Would love an offline mode for trail runs with no signal.",
        "Please add dark mode, the white screen is blinding at 5am.",
        "Wish I could export my runs as GPX files.",
        "Can you add strength training to the plans, not just running?",
        "Would be great to share routes with friends in the app.",
        "Please let me customise which stats show on the run screen.",
        "An option to set a different weekly goal for each week would be great.",
    ]),
}

MISC = [  # neutral/noise reviews that don't belong to a theme
    ("It's fine. Does what it says.", 3),
    ("Haven't used it much yet.", 3),
    ("Good app.", 4),
    ("Switched from another app, still getting used to it.", 3),
    ("ok", 3),
    ("Nice colors.", 4),
    ("Works on my phone.", 4),
    ("Five stars for the community challenges!", 5),
]

OPENERS = ["", "", "", "Long-time user here. ", "Update: ", "Honestly, ", "Ugh. ", "Quick note: "]
CLOSERS = ["", "", "", " Please fix.", " Otherwise great app.", " Thinking of switching.", " Thanks team!", ""]


def main() -> None:
    rows = []
    for _theme, (count, (lo, hi), phrasings) in THEMES.items():
        for i in range(count):
            rating = random.randint(lo, hi)
            text = phrasings[i % len(phrasings)]
            opener = random.choice(OPENERS)
            closer = random.choice(CLOSERS)
            if rating >= 4:  # keep tone consistent with rating
                opener = opener if opener in ("", "Long-time user here. ", "Honestly, ") else ""
                closer = closer if closer in ("", " Thanks team!") else ""
            rows.append((opener + text + closer, rating))
    rows.extend(MISC)
    random.shuffle(rows)

    start = date(2026, 7, 1)
    out = Path(__file__).resolve().parent.parent / "sample_data" / "stride_reviews.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "date", "source", "rating", "review_text"])
        for i, (text, rating) in enumerate(rows, start=1):
            d = start + timedelta(days=random.randint(0, 91))
            src = random.choice(["App Store", "Google Play"])
            w.writerow([f"R{i:04d}", d.isoformat(), src, rating, text])
    print(f"Wrote {len(rows)} reviews to {out}")


if __name__ == "__main__":
    main()
