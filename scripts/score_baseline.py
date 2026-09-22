"""Score free-text JSON letters from results/reaction_baseline.json vs expectations."""
import json
import sys

sys.path.insert(0, "src")

from flashbulljev.reaction import check_case, reaction_cases

base = json.load(open("results/reaction_baseline.json"))
by_id = {c["id"]: c for c in reaction_cases()}
ok = tot = 0
for row in base["rows"]:
    case = by_id[row["id"]]
    exp = dict(case["expect"])
    letter = row.get("json_letter")
    if letter is None:
        print(f"{row['id']}: no letter parsed (text={row.get('json_text', '')[:60]!r})")
        continue
    q = case["question"]
    t = q.get("type")
    if t in ("noul", "boolean"):
        pred = {"noul": 1.0 if letter == "A" else 0.0}
    elif t == "choice":
        opts = list(q["criteria"].keys())
        pred = {"choice": opts[ord(letter) - 65] if 0 <= ord(letter) - 65 < len(opts) else None}
    elif t == "score":
        pred = {"probabilities": {str(i): 1.0 if i == ord(letter) - 65 else 0.0 for i in range(3)}}
    else:
        continue
    tot += 1
    good = check_case(pred, exp)
    ok += good
    print(f"{row['id']}: letter={letter} -> {'OK' if good else 'KO'} (expect={exp})")
print(f"json accuracy: {ok}/{tot}={ok / max(tot, 1):.3f}")
