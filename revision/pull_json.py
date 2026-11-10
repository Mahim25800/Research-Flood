"""Take the last REVB64 payload from the session transcript and store it as results_rev/<name>.json."""
import base64, gzip, json, re, sys
t = open('/root/.claude/projects/-home-claude/cbb07236-8ea9-5607-9f3f-d2df2dc47063.jsonl').read()
m = re.findall(r'REVB64:([A-Za-z0-9+/=]{1000,}):END', t)
d = json.loads(gzip.decompress(base64.b64decode(m[-1])))
json.dump(d, open(f'results_rev/{sys.argv[1]}.json', 'w'))
print(len(m), {k: list(v) for k, v in d['summary'].items()})
