const User = require('../models/User');
const { extractResumeText, inferProfile } = require('../services/resumeService');
const { calculateATS } = require('../services/atsService');
const { isProfileComplete } = require('../services/profileService');
async function uploadResume(req,res){
 try {
  if(!req.file)return res.status(400).json({message:'Resume file is required.'});
  const user=await User.findById(req.userId); if(!user)return res.status(404).json({message:'User not found.'});
  const text=await extractResumeText(req.file.path,req.file.originalname); if(!text.trim())return res.status(422).json({message:'Could not extract text. Please use a text-based PDF or DOCX.'});
  const current=user.profile.toObject ? user.profile.toObject() : {...user.profile};
  const resumeMeta={fileName:req.file.originalname,fileUrl:`/uploads/${req.file.filename}`,uploadedAt:new Date()};
  const inferred=inferProfile(text,current,resumeMeta);
  Object.keys(inferred).forEach(k=>{if(k==='resume')return; user.profile[k]=inferred[k];});
  user.profile.resume=inferred.resume;
  user.profileCompleted=isProfileComplete(user.profile);
  await user.save();
  res.json({user:user.toJSON(),ats:calculateATS(text,user.profile)});
 } catch(e){console.error(e);res.status(500).json({message:e.message||'Resume processing failed.'});}
}
async function analysis(req,res){ const user=await User.findById(req.userId); if(!user)return res.status(404).json({message:'User not found.'}); const text=user.profile?.resume?.text||''; if(!text)return res.status(404).json({message:'No resume uploaded yet.'}); res.json({ats:calculateATS(text,user.profile),resume:user.profile.resume}); }
module.exports={uploadResume,analysis};
