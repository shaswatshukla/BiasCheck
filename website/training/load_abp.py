"""Load ABP locally. Never publish article text or the resulting CSV."""
import json, glob, sys, hashlib, tarfile, subprocess
from pathlib import Path
import pandas as pd

def load_abp(repo_dir,min_per_source=8,max_per_source=40,max_words=400,seed=0):
    rows=[]
    def records():
        if (Path(repo_dir)/'.git').exists():
            process=subprocess.Popen(['git','-C',repo_dir,'archive','--format=tar','HEAD','data/jsons'],stdout=subprocess.PIPE)
            with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith('.json'):
                        yield json.load(archive.extractfile(member))
            if process.wait()!=0: raise RuntimeError('Git snapshot stream failed')
        else:
            for f in sorted(glob.glob(f'{repo_dir}/data/jsons/*.json')):
                with open(f,encoding='utf-8') as stream: yield json.load(stream)
    for index,r in enumerate(records()):
        if index and index%10000==0: print(f'Loaded {index} article records',flush=True)
        if r.get('bias_text') not in ['left','center','right']: continue
        text=' '.join(str(r.get('content') or '').split()[:max_words])
        if len(text.split())<60 or not r.get('source'): continue
        rows.append(dict(text=text,source=r['source'].strip().lower(),label=r['bias_text'],topic=r.get('topic'),date=r.get('date'),title=r.get('title')))
    df=pd.DataFrame(rows)
    pure=df.groupby('source').label.nunique().eq(1)
    audit={'eligible_articles':len(df),'sources':int(df.source.nunique()),'articles_in_single_label_sources_fraction':float(df.source.isin(pure[pure].index).mean())}
    # Remove duplicate prefixes globally before sampling, including cross-source syndication.
    df['digest']=df.text.map(lambda t:hashlib.sha256(t.lower().encode()).hexdigest())
    df=df.drop_duplicates('digest')
    n=df.groupby('source').size(); df=df[df.source.isin(n[n>=min_per_source].index)]
    df=df.sample(frac=1,random_state=seed).groupby('source',sort=True).head(max_per_source).drop(columns='digest').reset_index(drop=True)
    audit.update(sample_articles=len(df),sample_sources=int(df.source.nunique()),labels=df.label.value_counts().to_dict(),source_labels=df.groupby('label').source.nunique().to_dict())
    return df,audit

if __name__=='__main__':
    df,audit=load_abp(sys.argv[1]);df.to_csv(sys.argv[2],index=False)
    Path(sys.argv[2]+'.audit.json').write_text(json.dumps(audit,indent=2)); print(json.dumps(audit),flush=True)
