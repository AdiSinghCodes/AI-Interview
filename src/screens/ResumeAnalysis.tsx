
import { useEffect, useState } from 'react';
import { api } from '../api/api';
import { useAuth } from '../context/AuthContext';

export default function ResumeAnalysis(){
 const {user,uploadResume}=useAuth();
 const [data,setData]=useState<any>(null),[loading,setLoading]=useState(true),[error,setError]=useState('');
 const load=async()=>{setLoading(true);setError('');try{const r=await api.resumeAnalysis();setData(r)}catch(e:any){setError(e.message)}finally{setLoading(false)}};
 useEffect(()=>{if(user?.profile?.resume?.text)load();else setLoading(false)},[user]);
 const upload=async(file?:File)=>{if(!file)return;setLoading(true);setError('');try{const r=await uploadResume(file);setData({ats:r.ats,resume:r.user.profile.resume})}catch(e:any){setError(e.message)}finally{setLoading(false)}};
 if(!user) return null;
 return <div style={{minHeight:'100%',background:'#FAF9F5',padding:'28px 24px 70px',color:'#172033'}}>
  <div style={{maxWidth:1050,margin:'0 auto'}}>
   <div style={{marginBottom:20}}><div style={{fontSize:11,fontWeight:800,color:'#3358E8',letterSpacing:'.08em'}}>VIVA · RESUME ANALYSIS</div><h1 style={{fontFamily:'Outfit',fontSize:30,margin:'6px 0'}}>ATS Resume Score</h1><p style={{color:'#667085',fontSize:13}}>The score is calculated by the backend from the resume stored in MongoDB. It refreshes whenever the server is running.</p></div>
   <section style={{background:'#fff',border:'1px solid #e4e7ec',borderRadius:14,padding:20,marginBottom:14}}>
    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',gap:15,flexWrap:'wrap'}}><div><b>{user.profile?.resume?.fileName||'No resume uploaded'}</b><div style={{fontSize:12,color:'#667085',marginTop:4}}>Upload a new version to recalculate the score.</div></div><label style={{background:'#3358E8',color:'#fff',padding:'10px 14px',borderRadius:9,fontWeight:800,fontSize:12,cursor:'pointer'}}>Upload Resume<input type="file" accept=".pdf,.doc,.docx,.txt" style={{display:'none'}} onChange={e=>upload(e.target.files?.[0])}/></label></div>
   </section>
   {loading&&<div style={card}>Calculating ATS score…</div>}
   {error&&<div style={{...card,color:'#b42318',background:'#fff0f0'}}>{error}</div>}
   {!loading&&!data&&!error&&<div style={card}>No resume found. Upload your resume to get an ATS score.</div>}
   {data&&<><section style={{...card,display:'flex',alignItems:'center',gap:24,flexWrap:'wrap'}}>
      <div style={{width:130,height:130,borderRadius:'50%',display:'grid',placeItems:'center',background:`conic-gradient(#3358E8 ${data.ats.score}%,#e8eaf0 0)`}}><div style={{width:102,height:102,borderRadius:'50%',background:'#fff',display:'grid',placeItems:'center',fontFamily:'Outfit',fontSize:30,fontWeight:800}}>{data.ats.score}</div></div>
      <div><h2 style={{fontFamily:'Outfit',margin:'0 0 5px'}}>ATS compatibility score</h2><p style={{color:'#667085',fontSize:13,margin:0}}>{data.ats.score>=80?'Strong resume structure.':data.ats.score>=60?'Good foundation with improvements needed.':'Several important ATS signals are missing.'}</p></div>
    </section>
    <section style={card}><h2 style={h2}>Checks</h2><div style={{display:'grid',gridTemplateColumns:'repeat(2,minmax(0,1fr))',gap:10}}>{Object.entries(data.ats.checks).map(([k,v]:any)=><div key={k} style={{padding:12,border:'1px solid #e4e7ec',borderRadius:9}}><b>{v?'✓':'○'} {k}</b><div style={{fontSize:11,color:'#667085',marginTop:3}}>{v?'Detected':'Consider adding or improving this section'}</div></div>)}</div></section>
    <section style={card}><h2 style={h2}>Suggested improvements</h2>{data.ats.improvements.length?<ul>{data.ats.improvements.map((x:string)=><li key={x} style={{marginBottom:8}}>{x}</li>)}</ul>:<p style={{color:'#16834D'}}>No major structural gaps detected.</p>}</section>
   </>}
  </div>
 </div>
}
const card:React.CSSProperties={background:'#fff',border:'1px solid #e4e7ec',borderRadius:14,padding:20,marginBottom:14};
const h2:React.CSSProperties={fontFamily:'Outfit',fontSize:18,margin:'0 0 14px'};
