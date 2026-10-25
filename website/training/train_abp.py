"""Grouped stacked baseline + source-held-out temperature calibration; exports weights only."""
import json, sys, hashlib, subprocess, platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score,log_loss
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from load_abp import load_abp
from newsbias import clean, side_features, US

LABELS=['left','center','right']
def grouped(d,k=3):
    for seed in range(40):
        folds=list(StratifiedGroupKFold(k,shuffle=True,random_state=seed).split(d.text,d.label,d.source))
        if all(len(set(d.label.iloc[tr]))==3 and len(set(d.label.iloc[te]))==3 for tr,te in folds): return folds
    raise ValueError('Insufficient class/source coverage for grouped folds')
def weights(d):
    w=1/d.source.map(d.source.value_counts()).to_numpy();return w/w.mean()
def text_fit(d):
    v=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_df=.5,max_features=10000,sublinear_tf=True,stop_words='english',token_pattern=r'(?u)\b[a-z][a-z0-9_]+\b')
    x=v.fit_transform(d.text);m=LogisticRegression(max_iter=1000,C=1,class_weight='balanced',solver='lbfgs').fit(x,d.label,sample_weight=weights(d));return v,m
def prob(vm,texts):
    v,m=vm;p=m.predict_proba(v.transform(texts));return p[:,[list(m.classes_).index(c) for c in LABELS]]
def features(p,texts):
    logs=np.log(p+1e-4);logs-=logs.mean(1,keepdims=True)
    sides=np.array([side_features(t,US) for t in texts])
    return np.column_stack([logs,np.zeros((len(p),3)),sides[:,1]-sides[:,0],sides[:,2],np.zeros(len(p))])
def fit_stack(d):
    p=np.zeros((len(d),3))
    for tr,te in grouped(d):p[te]=prob(text_fit(d.iloc[tr]),d.text.iloc[te])
    scale=StandardScaler().fit(features(p,d.text));meta=LogisticRegression(C=.5,class_weight='balanced',max_iter=1000).fit(scale.transform(features(p,d.text)),d.label,sample_weight=weights(d))
    return text_fit(d),scale,meta
def logits(model,d):
    vm,sc,m=model;p=prob(vm,d.text);z=m.decision_function(sc.transform(features(p,d.text)));return z[:,[list(m.classes_).index(c) for c in LABELS]],p
def calibrate(model,d):
    z,_=logits(model,d);y=np.array([LABELS.index(c) for c in d.label]);w=weights(d)
    return float(minimize_scalar(lambda t:log_loss(y,softmax(z/t,axis=1),labels=[0,1,2],sample_weight=w),bounds=(.3,8),method='bounded').x)
def partition(d):
    # Entire calibration outlets are held out from vocabulary, text classifier and stack.
    for tr,ca in grouped(d,4):
        try:grouped(d.iloc[tr].reset_index(drop=True));return d.iloc[tr].reset_index(drop=True),d.iloc[ca].reset_index(drop=True)
        except ValueError:continue
    raise ValueError('Not enough sources for nested grouped training/calibration')
def ece(p,y,w):
    conf=p.max(1);correct=(p.argmax(1)==y);out=0
    for lo in np.arange(0,1,.1):
        ix=(conf>lo)&(conf<=lo+.1)
        if ix.any():out+=w[ix].sum()/w.sum()*abs(np.average(correct[ix],weights=w[ix])-np.average(conf[ix],weights=w[ix]))
    return float(out)
def predict(model,temp,d):
    z,pt=logits(model,d);p=softmax(z/temp,axis=1);tone=np.array([np.tanh(side_features(t,US)[2]*2) for t in d.text]);dis=np.std(np.column_stack([pt[:,2]-pt[:,0],tone]),axis=1)
    return p,(p.max(1)>=.5)&(dis<=.9)
def main(repo):
    root=Path(__file__).resolve().parents[1];out=root/'.local';out.mkdir(exist_ok=True)
    d,audit=load_abp(repo);names=sorted(d.source.unique());d['text']=[clean(t,names) for t in d.text]
    audit['repository_commit']=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip();print(json.dumps(audit),flush=True)
    p=np.zeros((len(d),3));pt=np.zeros_like(p);answered=np.zeros(len(d),bool);folds=[]
    for k,(tr,te) in enumerate(grouped(d)):
        train=d.iloc[tr].reset_index(drop=True);test=d.iloc[te].reset_index(drop=True);core,cal=partition(train)
        print(f'Outer fold {k+1}: {core.source.nunique()} train / {cal.source.nunique()} calibration / {test.source.nunique()} test outlets',flush=True)
        model=fit_stack(core);temp=calibrate(model,cal);p[te],answered[te]=predict(model,temp,test);pt[te]=prob(text_fit(train),test.text)
        assert not(set(core.source)&set(cal.source) or set(train.source)&set(test.source))
        folds.append({'test_sources':sorted(test.source.unique()),'training_sources':sorted(core.source.unique()),'calibration_sources':sorted(cal.source.unique()),'temperature':temp})
    y=np.array([LABELS.index(c) for c in d.label]);w=weights(d);pred=p.argmax(1)
    metrics={'source_weighted_accuracy':float(accuracy_score(y,pred,sample_weight=w)),'source_weighted_macro_f1':float(f1_score(y,pred,average='macro',sample_weight=w)),'source_weighted_coverage':float(np.average(answered,weights=w)),'source_weighted_answered_accuracy':float(accuracy_score(y[answered],pred[answered],sample_weight=w[answered])) if answered.any() else None,'ece_10_bins_source_weighted':ece(p,y,w),'articles':len(d),'outlets':int(d.source.nunique())}
    # Leakage-gap diagnostic compares text-only random vs grouped CV, never promotes random score as deployment accuracy.
    random_p=np.zeros_like(p)
    for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(d.text,d.label):random_p[te]=prob(text_fit(d.iloc[tr]),d.text.iloc[te])
    metrics['text_random_macro_f1']=float(f1_score(y,random_p.argmax(1),average='macro'));metrics['text_grouped_macro_f1']=float(f1_score(y,pt.argmax(1),average='macro'));metrics['leakage_gap']=metrics['text_random_macro_f1']-metrics['text_grouped_macro_f1']
    print(json.dumps(metrics),flush=True)
    core,cal=partition(d);model=fit_stack(core);temp=calibrate(model,cal);v,m=model[0];sc,meta=model[1:];order=[list(m.classes_).index(c) for c in LABELS];mo=[list(meta.classes_).index(c) for c in LABELS]
    artifact={'version':1,'name':'ABP source-held-out US research baseline','labels':LABELS,'country':'United States','maxWords':400,'minimumConfidence':.5,'maximumDisagreement':.9,'sourceNames':names,'stopWords':sorted(ENGLISH_STOP_WORDS),'vocabulary':v.get_feature_names_out().tolist(),'idf':v.idf_.tolist(),'textCoef':m.coef_[order].tolist(),'textIntercept':m.intercept_[order].tolist(),'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'metaCoef':meta.coef_[mo].tolist(),'metaIntercept':meta.intercept_[mo].tolist(),'temperature':temp,'metrics':metrics,'trainingArticles':len(core),'trainingOutlets':int(core.source.nunique()),'calibrationOutlets':int(cal.source.nunique())}
    dest=root/'public/models';dest.mkdir(exist_ok=True,parents=True);path=dest/'abp-us-v1.json';path.write_text(json.dumps(artifact,separators=(',',':')),encoding='utf-8')
    card={'dataset':'https://github.com/ramybaly/Article-Bias-Prediction','paper':'https://aclanthology.org/2020.emnlp-main.404/','audit':audit,'metrics':metrics,'folds':folds,'final_training_sources':sorted(core.source.unique()),'final_calibration_sources':sorted(cal.source.unique()),'python':platform.python_version(),'sklearn':sklearn.__version__,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'limitations':['US-oriented research model; no India/UK validation.','Labels can act as outlet ratings; outlet independence is essential.','Source-weighted metrics treat each outlet equally; no article-level independence claim.','Exact 400-word prefix duplicates removed. Near duplicates and same-event overlap may remain.','No source priors, peers or LLM used. Text probabilities are cross-fitted by source; final temperature uses separate calibration outlets.','TF-IDF lexical associations are not proof of article bias or factual accuracy. Topic/date transfer is not evaluated.','Dataset article text and sampled CSV remain local and are not distributed.']}
    (dest/'abp-model-card.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
    # Synthetic parity fixtures only; never export publisher article text.
    fixtures=[]
    import pandas as pd
    for text in ['Congress said the government should expand healthcare and protect workers.','The government must reduce taxes and support a free market economy.','The committee met on Tuesday to discuss a proposed budget and release figures.']:
        row=pd.DataFrame({'text':[clean(text,names)]});pp,aa=predict(model,temp,row);fixtures.append({'text':text,'probabilities':pp[0].tolist(),'answered':bool(aa[0])})
    (root/'tests/abp-parity.json').write_text(json.dumps(fixtures,indent=2),encoding='utf-8'); print('Exported weights, model card and synthetic parity fixtures',flush=True)
if __name__=='__main__': main(sys.argv[1])
