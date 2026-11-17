"""Execute offline-copy safety cases using only disposable local repositories."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
SCRIPT=HERE/'COPY_TO_LAPTOP_OFFLINE.ps1'
assert SCRIPT.is_file(), 'The current fail-closed offline-copy implementation is missing'

def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT).strip()

def commit(repo,message):
    git(repo,'add','.')
    git(repo,'-c','user.name=Review test','-c','user.email=review@example.invalid','commit','-m',message)

results=[]
with tempfile.TemporaryDirectory(prefix='aerofoil-review-copy-tests-') as temporary:
    base=Path(temporary)
    for case in ['valid','tracked_dirty','untracked','wrong_branch','unpushed_commit','source_tampered','existing_destination','remote_ahead','escaping_manifest']:
        home=base/case;home.mkdir()
        remote=home/'remote.git';remote.mkdir();git(remote,'init','--bare')
        repo=home/'repo';repo.mkdir();git(repo,'init','-b','robust-v2-viability')
        (repo/'tracked.txt').write_text('review fixture',encoding='utf-8');commit(repo,'base')
        git(repo,'remote','add','origin',str(remote));git(repo,'push','-u','origin','robust-v2-viability')
        head=git(repo,'rev-parse','HEAD')
        source=home/'source';source.mkdir();(source/'evidence.txt').write_text('unchanged evidence',encoding='utf-8')
        payload=(source/'evidence.txt').read_bytes()
        manifest={'schema':'academic-review-release-v2','source_branch':'robust-v2-viability','source_commit_sha':head,'source_remote_sha':head,'files':[{'path':'evidence.txt','size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
        destination=home/'copied'
        if case=='tracked_dirty':(repo/'tracked.txt').write_text('local work',encoding='utf-8')
        if case=='untracked':(repo/'new.txt').write_text('local work',encoding='utf-8')
        if case=='wrong_branch':git(repo,'switch','-c','other-branch')
        if case=='unpushed_commit':
            (repo/'tracked.txt').write_text('unpublished work',encoding='utf-8');commit(repo,'local-only')
        if case=='source_tampered':(source/'evidence.txt').write_text('tampered',encoding='utf-8')
        if case=='existing_destination':destination.mkdir();(destination/'keep.txt').write_text('keep me',encoding='utf-8')
        if case=='remote_ahead':
            other=home/'other';git(home,'clone','-b','robust-v2-viability',str(remote),str(other))
            (other/'tracked.txt').write_text('remote work',encoding='utf-8');commit(other,'remote-only');git(other,'push')
        if case=='escaping_manifest':manifest['files'][0]['path']='../outside.txt'
        (source/'SHA256_MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
        before={'head':git(repo,'rev-parse','HEAD'),'branch':git(repo,'branch','--show-current'),'status':git(repo,'status','--porcelain=v1','--untracked-files=all')}
        result=subprocess.run(['powershell.exe','-NoProfile','-File',str(SCRIPT),'-Source',str(source),'-RepositoryPath',str(repo),'-Destination',str(destination)],text=True,capture_output=True,check=False)
        after={'head':git(repo,'rev-parse','HEAD'),'branch':git(repo,'branch','--show-current'),'status':git(repo,'status','--porcelain=v1','--untracked-files=all')}
        assert before==after,(case,'repository changed',before,after)
        assert (result.returncode==0)==(case=='valid'),(case,result.returncode,result.stdout,result.stderr)
        if case=='valid':assert (destination/'evidence.txt').read_bytes()==payload
        if case=='existing_destination':assert (destination/'keep.txt').read_text()=='keep me'
        results.append({'case':case,'expected':'PASS' if case=='valid' else 'REFUSED','exit_code':result.returncode,'repository_unchanged':before==after,'output':(result.stdout+result.stderr).strip()[-1600:]})
        print(case,'EXPECTED_OUTCOME_CONFIRMED',flush=True)
(HERE/'OFFLINE_COPY_TESTS.json').write_text(json.dumps({'cases':results,'all_passed':True},indent=2),encoding='utf-8')
print('OFFLINE_COPY_SAFETY',len(results),'of',len(results),flush=True)
