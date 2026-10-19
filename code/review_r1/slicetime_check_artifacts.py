"""Read-only final A7 manifest/output/inventory integrity verification."""
from slicetime_common import *


def main():
    counts=dict(manifests=0,input_hashes=0,source_hashes=0,output_hashes=0,inventory_hashes=0)
    for p in OUT.rglob('input_manifest.json'):
        q=json.loads(p.read_text());counts['manifests']+=1
        for path,r in q['inputs'].items():assert digest(path)==r['sha256'],path;counts['input_hashes']+=1
        for path,sha in q['sources'].items():assert digest(path)==sha,path;counts['source_hashes']+=1
        validation=p.with_name('validation.json');v=json.loads(validation.read_text());assert v['passed']
        for name,sha in v['output_sha256'].items():assert digest(p.parent/name)==sha,(p,name);counts['output_hashes']+=1
    table=pd.read_csv(OUT/'files_created.csv')
    for r in table.itertuples():assert Path(r.path).stat().st_size==r.bytes and digest(r.path)==r.sha256,r.path;counts['inventory_hashes']+=1
    print(json.dumps(dict(passed=True,**counts),indent=2),flush=True)

if __name__=='__main__':main()
