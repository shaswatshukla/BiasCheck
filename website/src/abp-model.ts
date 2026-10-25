import type {Assessment} from './types';
export type AbpModel={version:number;name:string;country:string;labels:string[];maxWords:number;minimumConfidence:number;maximumDisagreement:number;sourceNames:string[];stopWords:string[];vocabulary:string[];idf:number[];textCoef:number[][];textIntercept:number[];mean:number[];scale:number[];metaCoef:number[][];metaIntercept:number[];temperature:number;metrics:Record<string,number>;trainingArticles:number;trainingOutlets:number;calibrationOutlets:number};
export type TrainedResult={probabilities:number[];confidence:number;answered:boolean;matchedTerms:number;disagreement:number;metrics:Record<string,number>;name:string};
const esc=(s:string)=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const mean=(v:number[])=>v.length?v.reduce((a,b)=>a+b,0)/v.length:0;
const dot=(a:number[],b:number[])=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const softmax=(v:number[])=>{const e=v.map(x=>Math.exp(x-Math.max(...v))),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);};
export function cleanAbp(text:string,m:AbpModel){
 let t=text.trim().split(/\s+/u).slice(0,m.maxWords).join(' ');
 for(const name of m.sourceNames)if(name)t=t.replace(new RegExp(esc(name),'gi'),' ');
 return t.replace(/https?:\/\/\S+/g,' ').replace(/^\s*[A-Z][A-Z ,.'-]{2,40}\s*(\([^)]*\))?\s*[-—–:]\s*/u,' ').replace(/\b(photo|file photo|getty images?|ap photo|reuters)\b[^.\n]*/gi,' ').replace(/(follow us on|subscribe|sign up for|read more)[^.\n]*/gi,' ').replace(/\s+/g,' ').trim();
}
function sideFeatures(text:string){
 const low=text.toLowerCase(),n=Math.max(text.split(/\s+/).length,1);
 const count=(term:string)=>low.split(term).length-1;
 const left=['far-right','extremist','gutted','dark money','voter suppression','climate denier'].reduce((s,w)=>s+count(w),0)/n*1000;
 const right=['radical left','woke','mainstream media','illegal aliens','open borders','elites'].reduce((s,w)=>s+count(w),0)/n*1000;
 const la=['democrat','biden','harris','progressive'],ra=['republican','trump','gop','conservative'];
 const pos=new Set('good great success win praised welcomed strong boost growth relief hailed secure fair'.split(' ')),neg=new Set('bad crisis failure fail slammed attack corrupt scandal weak crackdown fraud harm threat unfair'.split(' '));
 const sl:number[]=[],sr:number[]=[];
 for(const s of text.split(/(?<=[.!?])\s+/u)){const x=s.toLowerCase(),a=la.some(k=>x.includes(k)),b=ra.some(k=>x.includes(k));const words=x.match(/[a-z']+/g)||[],p=words.filter(w=>pos.has(w)).length,q=words.filter(w=>neg.has(w)).length,tone=(p-q)/Math.max(p+q,1);if(a&&!b)sl.push(tone);else if(b&&!a)sr.push(tone);}
 return [left,right,mean(sr)-mean(sl)];
}
export function predictAbp(m:AbpModel,raw:string){
 const text=cleanAbp(raw,m),stop=new Set(m.stopWords),tokens=(text.toLowerCase().match(/(?<![\p{L}\p{N}_])[a-z][a-z0-9_]+(?![\p{L}\p{N}_])/gu)||[]).filter(w=>!stop.has(w));
 const index=new Map(m.vocabulary.map((v,i)=>[v,i])),counts=new Map<number,number>();
 const add=(term:string)=>{const i=index.get(term);if(i!==undefined)counts.set(i,(counts.get(i)||0)+1);};
 tokens.forEach((w,i)=>{add(w);if(i<tokens.length-1)add(w+' '+tokens[i+1]);});
 const tf=new Map<number,number>();let norm=0;for(const [i,count] of counts){const v=(1+Math.log(count))*m.idf[i];tf.set(i,v);norm+=v*v;}norm=Math.sqrt(norm)||1;for(const [i,v] of tf)tf.set(i,v/norm);
 const textLogits=m.textCoef.map((coef,c)=>m.textIntercept[c]+[...tf].reduce((sum,[i,v])=>sum+coef[i]*v,0));const pt=softmax(textLogits);
 const logs=pt.map(p=>Math.log(p+1e-4)),avg=mean(logs),side=sideFeatures(text);
 const row=[...logs.map(v=>v-avg),0,0,0,side[1]-side[0],side[2],0].map((v,i)=>(v-m.mean[i])/m.scale[i]);
 const probabilities=softmax(m.metaCoef.map((coef,c)=>(m.metaIntercept[c]+dot(coef,row))/m.temperature));
 const winner=probabilities.indexOf(Math.max(...probabilities)),confidence=probabilities[winner],votes=[pt[2]-pt[0],Math.tanh(side[2]*2)],av=mean(votes),disagreement=Math.sqrt(mean(votes.map(x=>(x-av)**2)));
 const answered=confidence>=m.minimumConfidence&&disagreement<=m.maximumDisagreement;
 const topTerms=[...tf].map(([i,v])=>({term:m.vocabulary[i],contribution:m.textCoef[winner][i]*v})).filter(x=>x.contribution>0).sort((a,b)=>b.contribution-a.contribution).slice(0,8).map(x=>x.term);
 return {probabilities,confidence,answered,matchedTerms:tf.size,disagreement,winner,topTerms};
}
let cached:Promise<AbpModel>|undefined;
export async function loadAbp(signal?:AbortSignal){
 if(!cached)cached=fetch('/models/abp-us-v1.json',{signal:signal?AbortSignal.any([signal,AbortSignal.timeout(20000)]):AbortSignal.timeout(20000)}).then(async r=>{if(!r.ok)throw Error('The research model could not be loaded.');const m=await r.json() as AbpModel;if(m.version!==1||m.labels.join(',')!=='left,center,right')throw Error('Unsupported research model.');return m;}).catch(e=>{cached=undefined;throw e;});return cached;
}
export function trainedAssessment(model:AbpModel,text:string,base:Assessment):Assessment{
 const p=predictAbp(model,text),leanings=['Left leaning','Centrist','Right leaning'];const result:TrainedResult={...p,metrics:model.metrics,name:model.name};
 const quotes=text.split(/(?<=[.!?])\s+|\n+/u).map(s=>s.trim()).filter(s=>s.length&&s.length<=300);
 const evidence:Assessment['evidence']=[];for(const term of p.topTerms){const sentence=quotes.find(s=>new RegExp('\\b'+esc(term)+'\\b','i').test(s));if(sentence&&!evidence.some(e=>e.quote===sentence))evidence.push({dimension:'Learned lexical association',direction:'Unclear',quote:sentence,explanation:`“${term}” contributed to the text classifier’s ${model.labels[p.winner]} association. This is a corpus-level association, not proof that this sentence endorses an ideology.`});if(evidence.length>=4)break;}
 return {...base,model:'ABP US stacked research model v1',trained:result,leaning:p.answered?leanings[p.winner]:'Uncertain',evidence_strength:'Low',summary:`US research model: ${p.answered?leanings[p.winner]:'abstained from a label'}. Trained on ABP articles with outlet-grouped validation and separate source-held-out temperature calibration. Scores reflect dataset label associations, not verified bias. Only the first ${model.maxWords} words are analysed.`,evidence:[...evidence,...base.evidence].slice(0,10),limitations:[...base.limitations.filter(s=>!s.includes('No labelled dataset')&&!s.includes('Percentages are an uncalibrated')),'US-only experimental classifier; not validated for Indian or UK political context.','ABP labels are strongly tied to outlets; a held-out-source test reduces publisher leakage but cannot establish objective article ideology.','Temperature calibration uses separate outlets. It does not guarantee probabilities transfer to new dates, topics or publishers.','Source priors and peer comparisons are disabled. Sensationalism remains a separate heuristic. Lexical attributions explain the base text model, not the full stacked prediction.']};
}
