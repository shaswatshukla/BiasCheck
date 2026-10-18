import http from 'node:http';
import { spawn } from 'node:child_process';
import { register } from 'tsx/esm/api';
register();
const {default:handler}=await import('../netlify/functions/api.ts');
const server=http.createServer(async(req,res)=>{try{let size=0;const chunks=[];for await(const chunk of req){size+=chunk.length;if(size>150000){res.writeHead(413);res.end('Request too large');return;}chunks.push(chunk);}const request=new Request(`http://127.0.0.1:5174${req.url}`,{method:req.method,headers:req.headers,...(['GET','HEAD'].includes(req.method)?{}:{body:Buffer.concat(chunks)})});const response=await handler(request);res.writeHead(response.status,Object.fromEntries(response.headers));res.end(Buffer.from(await response.arrayBuffer()));}catch{res.writeHead(500);res.end(JSON.stringify({error:'Local request failed.'}));}});
server.listen(8889,'127.0.0.1');
const child=spawn(process.execPath,['node_modules/vite/bin/vite.js'],{stdio:'inherit'});
function stop(){server.close();child.kill();}
process.on('SIGINT',stop);process.on('SIGTERM',stop);child.on('close',()=>{server.close();});
