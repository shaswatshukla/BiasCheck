import type { Assessment } from './types';

// Hand-authored policy-position rules. These are inspectable heuristics, not a trained classifier.
const rules: {direction:'Left'|'Right'|'Centrist';name:string;pattern:RegExp;india?:boolean}[]=[
 {direction:'Left',name:'Redistribution',pattern:/\b(?:raise|increase|higher)\b.{0,45}\b(?:taxes? on (?:the )?(?:rich|wealthy)|wealth tax|corporate taxes?)\b|\b(?:tax the rich|redistribut(?:e|ion) wealth)\b/i},
 {direction:'Left',name:'Public provision',pattern:/\b(?:universal|free|publicly funded)\s+(?:healthcare|health care|education)|\b(?:expand|strengthen|increase)\b.{0,35}\b(?:welfare|social safety net|public spending)\b/i},
 {direction:'Left',name:'Labour protection',pattern:/\b(?:strengthen|protect|expand)\b.{0,30}\b(?:workers.? rights|trade unions|collective bargaining)|\b(?:raise|increase)\s+(?:the )?minimum wage\b/i},
 {direction:'Left',name:'Social equality',pattern:/\b(?:protect|support|defend|expand|guarantee)\b.{0,35}\b(?:minority rights|LGBTQ\+? rights|reproductive rights|abortion rights|same.sex marriage)\b/i},
 {direction:'Right',name:'Market-led policy',pattern:/\b(?:cut|reduce|lower)\b.{0,30}\b(?:corporate taxes|taxes on businesses)|\b(?:privati[sz]ation|free.market reforms|deregulation)\b/i},
 {direction:'Right',name:'Smaller state',pattern:/\b(?:cut|reduce|limit|slash)\b.{0,35}\b(?:welfare spending|government spending|public spending|size of government)|\b(?:smaller government|fiscal conservatism)\b/i},
 {direction:'Right',name:'Traditional social policy',pattern:/\b(?:ban|restrict|oppose)\b.{0,30}\b(?:abortion|same.sex marriage)|\b(?:defend|restore|protect)\b.{0,20}\btraditional (?:family|values)\b/i},
 {direction:'Right',name:'Immigration restriction',pattern:/\b(?:restrict|reduce|curb|stop)\b.{0,25}\bimmigration\b|\b(?:stricter|tighter)\s+(?:border|immigration) controls\b/i},
 {direction:'Left',name:'Secular constitutional policy',pattern:/\b(?:defend|protect|uphold|strengthen)\b.{0,30}\b(?:secularism|secular constitution)|\b(?:oppose|reject)\b.{0,25}\b(?:Hindu nationalism|Hindutva)\b/i,india:true},
 {direction:'Right',name:'Religious-nationalist policy',pattern:/\b(?:support|embrace|defend|promote)\b.{0,25}\b(?:Hindu nationalism|Hindutva|Hindu rashtra)\b/i,india:true},
 {direction:'Centrist',name:'Explicit middle-ground policy',pattern:/\b(?:middle.ground|cross.party compromise|bipartisan compromise)|\bbalance\b.{0,35}\b(?:market|growth)\b.{0,35}\b(?:welfare|social protection)\b/i},
];
export function localAssessment(text:string,country:string):Assessment{
 const supported=['India','United States','United Kingdom'].includes(country);
 const english=(text.match(/[A-Za-z]/g)||[]).length/Math.max(1,(text.match(/\p{L}/gu)||[]).length)>.7;
 const evidence:Assessment['evidence']=[];const seen=new Set<string>();
 if(supported&&english)for(const raw of text.replace(/“[^”]*”|"[^"]*"/gu," ").split(/(?<=[.!?])\s+|\n+/u)){
  const sentence=raw.trim();if(!sentence||sentence.length>300)continue;
  // Quoted speech, reported claims, negation and criticism are too ambiguous for these rules.
  if(/["“”]/u.test(sentence)||/\b(?:said|says|according to|claim\w*|argu\w*|told|not|never|cannot|no longer|against|reject\w*|oppos\w*|criticis\w*|criticiz\w*|condemn\w*|harmful|dangerous|disastrous|failed)\b/i.test(sentence))continue;
  if(!/\b(?:should|must|need to|ought to|we support|we advocate|we favour|we favor|we call for)\b/i.test(sentence))continue;
  for(const rule of rules){if(rule.india&&country!=='India'||seen.has(rule.name)||!rule.pattern.test(sentence))continue;seen.add(rule.name);evidence.push({dimension:'Policy stance',direction:rule.direction,quote:sentence,explanation:`Matched the “${rule.name}” rule in an explicit advocacy sentence. This is a preliminary policy signal; a phrase rule cannot establish author intent.`});}
 }
 const left=evidence.filter(e=>e.direction==='Left').length,right=evidence.filter(e=>e.direction==='Right').length,center=evidence.filter(e=>e.direction==='Centrist').length;
 const leaning=left&&right?'Mixed':left>=2?'Left leaning':right>=2?'Right leaning':center&& !left&&!right?'Centrist':'Uncertain';
 const reason=!supported?'This political context is not covered by the local rules.':!english?'The local rules support English text only.':leaning==='Uncertain'?'Too few distinct, explicit policy positions matched. Topic mentions alone do not establish political leaning.':`The rules found ${left} left-associated, ${right} right-associated and ${center} middle-ground policy signals.`;
 return {leaning,evidence_strength:'Low',summary:`Preliminary local assessment. ${reason}`,evidence:evidence.slice(0,6),alternative_reading:'A policy position can appear in reporting, irony or criticism. Review the matched sentences in their full context; this result may misread the writer’s position.',limitations:['Rule-based English screening, not a validated political-bias classifier. It cannot reliably interpret implicit framing, sarcasm or complex negation.','Only explicit advocacy is scored. Quoted and attributed statements are skipped. Unmatched wording can produce Uncertain even when a stance exists.','Left/right associations vary by country and issue. Government criticism, reporter identity and outlet reputation are not scored.','All evidence strength is Low. Use optional Gemini analysis for a broader contextual reading.'],model:'Local policy rules v1',analyzed_at:new Date().toISOString()};
}
