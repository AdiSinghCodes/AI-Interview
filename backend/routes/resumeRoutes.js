const router=require('express').Router(); const multer=require('multer'); const path=require('path'); const fs=require('fs'); const c=require('../controllers/resumeController'); const {requireAuth}=require('../middleware/authMiddleware');
const dir=path.join(__dirname,'..','uploads'); fs.mkdirSync(dir,{recursive:true});
const MAX_RESUME_SIZE_MB=20;
const storage=multer.diskStorage({destination:dir,filename:(_r,f,cb)=>cb(null,`${Date.now()}-${f.originalname.replace(/[^a-zA-Z0-9._-]/g,'_')}`)});
const upload=multer({storage,limits:{fileSize:MAX_RESUME_SIZE_MB*1024*1024},fileFilter:(_r,f,cb)=>{const ok=['.pdf','.doc','.docx','.txt'].includes(path.extname(f.originalname).toLowerCase()); cb(ok?null:new Error('Only PDF, DOC, DOCX and TXT resumes are supported.'),ok);}});
function handleUpload(req,res,next){
  upload.single('resume')(req,res,(err)=>{
    if(!err) return next();
    if(err.code==='LIMIT_FILE_SIZE') return res.status(413).json({message:`Resume file exceeds the ${MAX_RESUME_SIZE_MB}MB limit. Please upload a smaller PDF, DOCX or TXT.`});
    return res.status(400).json({message: err.message || 'Resume upload failed.'});
  });
}
router.post('/upload',requireAuth,handleUpload,c.uploadResume); router.get('/analysis',requireAuth,c.analysis); router.post('/analysis',requireAuth,c.analysisFor); module.exports=router;
