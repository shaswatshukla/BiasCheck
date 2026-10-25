import { localAssessment } from './local-analysis';
import type { Assessment } from './types';

export type Signals={axis:string;leftName:string;rightName:string;words:number;leftPer1k:number;rightPer1k:number;entityTone:number;leftVoices:number;rightVoices:number;sensationalism:number;leftShare:number|null;rightShare:number|null;disagreement:boolean;status:string};
const contexts={
 India:{axis:'Opposition / government alignment · not ideological left/right',leftName:'Opposition-associated',rightName:'Government-associated',left:['authoritarian','majoritarian','crony capitalism','undeclared emergency','witch-hunt'],right:['appeasement','tukde','anti-national','urban naxal','vote bank politics'],la:['congress','rahul gandhi','aap','kejriwal','opposition'],ra:['bjp','modi','amit shah','nda','government']},
 'United States':{axis:'US left / right framing',leftName:'Left-associated',rightName:'Right-associated',left:['far-right','extremist','gutted','dark money','voter suppression','climate denier'],right:['radical left','woke','mainstream media','illegal aliens','open borders','elites'],la:['democrat','biden','harris','progressive'],ra:['republican','trump','gop','conservative']},
};
const pos=new Set('good great success win praised welcomed strong boost growth relief hailed secure fair'.split(' '));
const neg=new Set('bad crisis failure fail slammed attack corrupt scandal weak crackdown fraud harm threat unfair'.split(' '));
const escape=(s:string)=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const matches=(s:string,term:string)=>[...s.matchAll(new RegExp('\\b'+escape(term)+'\\b','gi'))].length;
const sentences=(s:string)=>s.split(/(?<=[.!?])\s+|\n+/u).filter(s=>s.trim());
const mean=(v:number[])=>v.length?v.reduce((a,b)=>a+b,0)/v.length:0;
export function cleanSignalText(text:string,source=''){
 let t=source?text.replace(new RegExp(escape(source),'gi'),' '):text;
 return t.replace(/https?:\/\/\S+/g,' ').replace(/^\s*[A-Z][A-Z ,.'-]{2,40}\s*(\([^)]*\))?\s*[-—–:]\s*/u,' ').replace(/\b(?:photo|file photo|getty images?|ap photo|reuters)\b[^.\n]*/gi,' ').replace(/(?:follow us on|subscribe|sign up for|read more)[^.\n]*/gi,' ').replace(/\s+/g,' ').trim();
}
export function multiSignalAssessment(text:string,country:string,source=''):Assessment{
 const base=localAssessment(text,country);const ctx=contexts[country as keyof typeof contexts];
 const cleaned=cleanSignalText(text,source),words=Math.max(cleaned.split(/\s+/).filter(Boolean).length,1),sents=sentences(cleaned);
 const english=(cleaned.match(/[a-z]/gi)||[]).length/Math.max((cleaned.match(/\p{L}/gu)||[]).length,1)>.7;
 if(!ctx||!english)return {...base,model:'Local multi-signal screening v1',limitations:[...base.limitations,'Multi-signal seeds support English in India and the US only. No trained model is loaded.']};
 let lv=0,rv=0;const lt:number[]=[],rt:number[]=[];
 for(const s of sents){const a=ctx.la.some(k=>matches(s,k)),b=ctx.ra.some(k=>matches(s,k));const tokens=s.toLowerCase().match(/[a-z']+/g)||[];const p=tokens.filter(w=>pos.has(w)).length,n=tokens.filter(w=>neg.has(w)).length;const tone=(p-n)/Math.max(p+n,1);if(a&&!b)lt.push(tone);if(b&&!a)rt.push(tone);if(/["“]/u.test(s)||/\b(?:said|says)\b/i.test(s)){lv+=Number(a);rv+=Number(b);}}
 const leftCount=ctx.left.reduce((n,w)=>n+matches(cleaned,w),0),rightCount=ctx.right.reduce((n,w)=>n+matches(cleaned,w),0);
 const tone=mean(rt)-mean(lt);
 // Fixed heuristic fusion only; this is not the Python model's learned/calibrated stack.
 const lexTotal=leftCount+rightCount;const lexical=lexTotal?(rightCount-leftCount)/lexTotal:0;
 const toneVote=Math.tanh(tone*2);const hasTone=lt.some(v=>v!==0)||rt.some(v=>v!==0);
 const conflict=lexTotal>0&&hasTone&&lexical*toneVote<0;
 const score=(lexical*(lexTotal?1:0)+toneVote*(hasTone?1:0))/Math.max(Number(lexTotal>0)+Number(hasTone),1);
 const hasSignal=lexTotal>0||hasTone;
 const leftShare=hasSignal?Math.round((1-score)/2*100):null;
 const tokens=cleaned.split(/\s+/);const caps=tokens.filter(w=>w.length>3&&w===w.toUpperCase()&&/[A-Z]/.test(w)).length/words;
 const excl=(cleaned.match(/!/g)||[]).length/Math.max(sents.length,1);
 const sup=(cleaned.match(/\b(shocking|devastating|explosive|bombshell|slams|destroys|stunning|outrage|meltdown|unbelievable|massive|historic)\b/gi)||[]).length/words*100;
 const click=Number(/you won't believe|what happens next|here's why|this is why/i.test(cleaned));
 const attrib=(cleaned.match(/\b(said|according to|told|reported|data|study|figures)\b/gi)||[]).length/words*100;
 const sensationalism=1/(1+Math.exp(-(4*caps+.5*excl+.3*sup+.5*click-.1*attrib-.3)*3));
 const signals:Signals={axis:ctx.axis,leftName:ctx.leftName,rightName:ctx.rightName,words,leftPer1k:leftCount/words*1000,rightPer1k:rightCount/words*1000,entityTone:tone,leftVoices:lv,rightVoices:rv,sensationalism,leftShare,rightShare:leftShare===null?null:100-leftShare,disagreement:conflict,status:'Heuristic signals only · trained classifier not loaded'};
 const evidence=[...base.evidence];const used=new Set<string>();
 for(const s of sentences(text)){if(s.length>300)continue;const l=ctx.left.filter(w=>matches(s,w)),r=ctx.right.filter(w=>matches(s,w));if((l.length||r.length)&&!used.has(s)){used.add(s);evidence.push({dimension:'Language',direction:'Unclear',quote:s.trim(),explanation:`Seed wording: ${[...l,...r].join(', ')}. A wording association can occur in a quotation, criticism or neutral reporting; it is not proof of endorsement.`});}}
 return {...base,model:'Local multi-signal screening v1',signals,evidence:evidence.slice(0,10),summary:`Free multi-signal screening. ${hasSignal?'Seed wording and entity-tone signals were found.':'No directional seed or entity-tone signal was found.'} ${conflict?'The two signals disagree. ':''}${country==='India'?'The displayed axis measures opposition/government associations, not ideological left/right. ':''}The policy assessment remains ${base.leaning.toLowerCase()}.`,limitations:[...base.limitations,'Ported signals from the supplied newsbias.py. India actor/word seeds are placeholders; associations are not ground-truth ideology.','No labelled dataset or fitted weights were supplied: TF-IDF/LR, learned stacking, calibration, source priors and peer comparisons are not active.','Percentages are an uncalibrated heuristic balance of lexical and entity-tone signals. Lexicon sentiment can misread attribution, negation and target scope. Quoted-voice counts do not influence the score.','Sensationalism is a separate, uncalibrated style index—not reliability, factual accuracy or political bias.']};
}
