"""Bounded primary bibliographic metadata audit using Crossref's public API.

Only already-public DOI strings are sent. No manuscript, dataset, credentials,
private paths or author correspondence are uploaded. Stop on 403/429; no retry.
"""
from pathlib import Path
import datetime
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent/'REFERENCES_PRIMARY'
OUT.mkdir(exist_ok=True)
text = (ROOT/'paper/manuscript.md').read_text(encoding='utf-8')
entries = []
for line in text.splitlines():
    match = re.match(r'^\[(\d+)\]\s+(.*)',line)
    if match:
        doi = re.search(r'10\.\d{4,9}/[^\s]+',match.group(2))
        entries.append({'number':int(match.group(1)),'original_entry':match.group(2),'doi':doi.group(0).rstrip('.,);') if doi else None})
if len(entries)!=28: raise RuntimeError(f'Expected original 28 references, saw {len(entries)}')
rows = []
blocked = False
for entry in entries:
    row = dict(entry,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if not entry['doi']:
        row['status'] = 'NO_DOI_WEB_PRIMARY_CHECK_REQUIRED'
    elif blocked:
        row['status'] = 'NOT_REQUESTED_AFTER_ACCESS_LIMIT'
    else:
        url = 'https://api.crossref.org/works/'+urllib.parse.quote(entry['doi'],safe='')
        row['primary_url'] = url
        try:
            request = urllib.request.Request(url,headers={'User-Agent':'RobustV2AcademicReferenceAudit/1.0','Accept':'application/json'})
            with urllib.request.urlopen(request,timeout=25) as response:
                data = response.read()
                row['http_status'] = response.status
            metadata = json.loads(data)['message']
            (OUT/f"reference_{entry['number']:02d}_crossref.json").write_bytes(data)
            row.update(status='PRIMARY_METADATA_VERIFIED',metadata_sha256=hashlib.sha256(data).hexdigest(),canonical_doi=metadata.get('DOI'),title=metadata.get('title'),authors=[{'given':a.get('given'),'family':a.get('family')} for a in metadata.get('author',[])],venue=metadata.get('container-title'),volume=metadata.get('volume'),issue=metadata.get('issue'),page=metadata.get('page'),article_number=metadata.get('article-number'),published=metadata.get('published'),published_print=metadata.get('published-print'),published_online=metadata.get('published-online'),type=metadata.get('type'))
        except urllib.error.HTTPError as exc:
            row.update(status='PRIMARY_METADATA_REQUEST_FAILED',http_status=exc.code,error=str(exc))
            blocked = exc.code in (403,429)
        except (urllib.error.URLError,TimeoutError,ValueError) as exc:
            row.update(status='PRIMARY_METADATA_REQUEST_FAILED',error=str(exc))
        time.sleep(1)
    rows.append(row)
    (OUT/'CROSSREF_LEDGER.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:row.get(k) for k in ['number','doi','status','title','http_status']},ensure_ascii=False),flush=True)
print('PRIMARY_METADATA_AUDIT_FINISHED',flush=True)
