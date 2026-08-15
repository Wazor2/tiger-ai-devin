import json
import sys

r = json.load(open("report/evaluation_report.json"))
keys = sys.argv[1:] or list(r["suites"])
for k in keys:
    s = r["suites"].get(k, {})
    if not s.get("available"):
        print(f"{k}: {s.get('error')}")
        print(s.get("traceback", "")[-500:])
    else:
        print(f"{k}: OK")
    print("---")
