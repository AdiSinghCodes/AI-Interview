const fs = require('fs');
const path = require('path');
const mammoth = require('mammoth');
const pdfParse = require('pdf-parse');
const { groqChatJSON } = require('./groqClient');

const PROFILE_FIELDS = [
  'fullName', 'phone', 'location', 'targetRole', 'experienceLevel', 'yearsExperience',
  'currentRole', 'summary', 'skills', 'degree', 'university', 'graduationYear',
  'linkedin', 'github', 'portfolio',
];

async function extractResumeText(filePath, originalName) {
  const ext = path.extname(originalName).toLowerCase();
  if (ext === '.txt') return fs.readFileSync(filePath, 'utf8');
  if (ext === '.pdf') return (await pdfParse(fs.readFileSync(filePath))).text || '';
  if (ext === '.docx') return (await mammoth.extractRawText({ path: filePath })).value || '';
  return '';
}

async function extractProfileWithLLM(rawText) {
  const text = rawText.slice(0, 14000);
  const systemPrompt =
    'You are a precise resume parser. Read the resume text and extract structured candidate data. ' +
    'Only use information present in the text - never invent facts. Respond with a single JSON object and nothing else.';
  const userPrompt =
    `Resume text:\n"""\n${text}\n"""\n\n` +
    'Return a JSON object with exactly these keys:\n' +
    '- fullName\n- phone\n- location (city, state/country)\n' +
    "- targetRole (best-fit job title based on the candidate's skills/experience)\n" +
    '- experienceLevel (one of: "Student", "Entry Level", "Mid Level", "Senior", "Lead/Manager")\n' +
    '- yearsExperience (approximate total years of professional experience as a plain number string, e.g. "2")\n' +
    '- currentRole (most recent job title, or current degree program if a student)\n' +
    '- summary (a concise 2-3 sentence professional summary written in third person)\n' +
    '- skills (comma-separated list of technical and professional skills actually mentioned)\n' +
    '- degree (highest degree, e.g. "B.Tech in Computer Science")\n' +
    '- university (school/university name)\n' +
    '- graduationYear (4-digit year as a string)\n' +
    '- linkedin (full LinkedIn profile URL if present)\n' +
    '- github (full GitHub profile URL if present)\n' +
    '- portfolio (personal website/portfolio URL if present)\n\n' +
    'Use an empty string "" for any field that cannot be found. Do not wrap the JSON in markdown.';

  const parsed = await groqChatJSON({
    systemPrompt,
    userPrompt,
    temperature: 0.1,
    maxTokens: 800,
    label: 'Resume LLM extraction',
  });
  if (!parsed) return null;

  const clean = {};
  PROFILE_FIELDS.forEach((key) => {
    const value = parsed[key];
    clean[key] = typeof value === 'string' ? value.trim() : value != null ? String(value).trim() : '';
  });
  return clean;
}

function regexExtractProfile(text) {
  const normalized = text.replace(/\s+/g, ' ').trim();
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);

  const phone = normalized.match(/(?:\+?\d[\d\s().-]{8,}\d)/)?.[0]?.trim() || '';
  const linkedin = normalized.match(/https?:\/\/(www\.)?linkedin\.com\/[A-Za-z0-9\-_/%]+/i)?.[0] || '';
  const github = normalized.match(/https?:\/\/(www\.)?github\.com\/[A-Za-z0-9\-_/%]+/i)?.[0] || '';
  const portfolioMatch = normalized.match(
    /https?:\/\/(?!(?:www\.)?(?:linkedin|github)\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:\/[A-Za-z0-9\-_./%?=&]*)?/i
  );
  const portfolio = portfolioMatch ? portfolioMatch[0] : '';

  const skillsList = [
    'JavaScript', 'TypeScript', 'Python', 'Java', 'C++', 'C#', 'React', 'Next.js', 'Node.js', 'Express',
    'MongoDB', 'PostgreSQL', 'MySQL', 'SQL', 'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Git', 'GitHub',
    'HTML', 'CSS', 'Tailwind', 'Flask', 'Django', 'FastAPI', 'TensorFlow', 'PyTorch', 'Machine Learning',
    'Artificial Intelligence', 'NLP', 'Computer Vision', 'DevOps', 'Linux', 'REST APIs', 'GraphQL', 'Redis',
    'Go', 'Rust', 'Swift', 'Kotlin', 'Scala', 'R', 'Spring', 'Spring Boot', '.NET', 'Angular', 'Vue',
    'Firebase', 'Jenkins', 'CI/CD', 'Terraform', 'Ansible', 'Pandas', 'NumPy', 'Scikit-learn',
  ];
  const foundSkills = skillsList.filter((s) =>
    new RegExp(`(^|\\W)${s.replace(/[.+#]/g, '\\$&')}(?=\\W|$)`, 'i').test(normalized)
  );

  const roles = [
    'Software Engineer', 'Frontend Engineer', 'Backend Engineer', 'Full Stack Developer', 'React Developer',
    'Node.js Developer', 'Python Developer', 'DevOps Engineer', 'Cloud Engineer', 'Data Scientist',
    'ML Engineer', 'AI Engineer', 'Data Analyst', 'Product Manager', 'Business Analyst', 'UI Designer',
    'UX Designer', 'QA Engineer', 'Mobile Developer',
  ];
  const targetRole = roles.find((r) => normalized.toLowerCase().includes(r.toLowerCase())) || '';

  const years = [...normalized.matchAll(/\b(19|20)\d{2}\b/g)].map((m) => Number(m[0])).filter((y) => y >= 1950 && y <= 2035);
  const gradContextMatch = normalized.match(/(?:graduat(?:ed|ion)?|class of|expected)[^\d]{0,20}((?:19|20)\d{2})/i);
  const graduationYear = gradContextMatch ? gradContextMatch[1] : years.length ? String(Math.max(...years)) : '';

  const nameCandidate = lines[0] || '';
  const fullName =
    nameCandidate && !/@|\d|linkedin|github|resume|curriculum/i.test(nameCandidate) && nameCandidate.length < 70
      ? nameCandidate
      : '';

  const degreeMatch = normalized.match(
    /(?:B\.?\s?Tech|B\.?\s?E\.?|B\.?\s?S\.?|B\.?\s?C\.?A|M\.?\s?Tech|M\.?\s?E\.?|M\.?\s?C\.?A|M\.?\s?B\.?A|Master(?:'s)?(?: of [A-Za-z]+)?|Bachelor(?:'s)?(?: of [A-Za-z]+)?)[^,.\n]{0,60}/i
  );
  const degree = degreeMatch ? degreeMatch[0].trim() : '';

  const universityMatch = normalized.match(/[A-Z][A-Za-z.&' ]{2,60}(?:University|Institute of Technology|Institute|College)\b/);
  const university = universityMatch ? universityMatch[0].trim() : '';

  const locationLine =
    lines.slice(0, 6).find((l) => /,/.test(l) && !/@/.test(l) && !/\d{4,}/.test(l) && l.length < 60 && !/linkedin|github/i.test(l)) || '';

  const experienceYearsMatch = normalized.match(/(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:professional\s+|work\s+)?experience/i);
  const yearsExperience = experienceYearsMatch ? experienceYearsMatch[1] : '';
  const experienceLevel = yearsExperience
    ? Number(yearsExperience) >= 6
      ? 'Lead/Manager'
      : Number(yearsExperience) >= 3
        ? 'Senior'
        : Number(yearsExperience) >= 1
          ? 'Mid Level'
          : 'Entry Level'
    : /intern|student|fresher/i.test(normalized)
      ? 'Student'
      : '';

  const currentRoleMatch = normalized.match(new RegExp(`(${roles.join('|')})\\s+(?:at|@|-)\\s+[A-Z][\\w&.,' ]{2,40}`, 'i'));
  const currentRole = currentRoleMatch ? currentRoleMatch[0].split(/\s+(?:at|@|-)\s+/i)[0].trim() : '';

  return {
    fullName,
    phone,
    location: locationLine,
    targetRole,
    experienceLevel,
    yearsExperience,
    currentRole,
    summary: normalized.slice(0, 600),
    skills: foundSkills.join(', '),
    degree,
    university,
    graduationYear,
    linkedin,
    github,
    portfolio,
  };
}

async function inferProfile(text, current = {}, resumeMeta = {}) {
  const llmProfile = await extractProfileWithLLM(text);
  const regexProfile = regexExtractProfile(text);
  const merged = {};
  PROFILE_FIELDS.forEach((key) => {
    const extracted = (llmProfile && llmProfile[key]) || regexProfile[key] || '';
    merged[key] = current[key] || extracted;
  });
  return {
    ...current,
    ...merged,
    resume: { ...(current.resume || {}), ...resumeMeta, text: text.replace(/\s+/g, ' ').trim() },
  };
}

module.exports = { extractResumeText, inferProfile };
