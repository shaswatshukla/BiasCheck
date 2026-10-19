import {mkdir,writeFile,readFile} from 'node:fs/promises';
await mkdir('public/fonts',{recursive:true});
const response=await fetch('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;450;500;550;600;650;700&family=Libre+Caslon+Display&display=swap');
if(!response.ok)throw Error('Font stylesheet unavailable');
let css=await response.text();let i=0;
for(const url of new Set([...css.matchAll(/url\((https:\/\/fonts\.gstatic\.com\/[^)]+)\)/g)].map(m=>m[1]))){const r=await fetch(url);if(!r.ok)throw Error('Font unavailable');const extension=new URL(url).pathname.split('.').at(-1);const path=`/fonts/font-${i++}.${extension}`;await writeFile(`public${path}`,Buffer.from(await r.arrayBuffer()));css=css.replaceAll(url,path);}
const filename='src/styles.css';const source=await readFile(filename,'utf8');await writeFile(filename,source.replace(/^@import[^\n]+\n/,css+'\n'));console.log(`Self-hosted ${i} font files.`);
