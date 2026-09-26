const { groqChatJSON } = require('./groqClient');

function heuristicATS(text = '', profile = {}) {
  const content = `${text} ${profile.targetRole || ''} ${profile.skills || ''}`.toLowerCase();
  const checks = {
    contact: /@/.test(content) && /\d{7,}/.test(content),
    skills: /(skills|technical skills|technologies)/.test(content),
    experience: /(experience|employment|work history)/.test(content),
    education: /(education|university|college|b\.?tech|bachelor|master)/.test(content),
    projects: /(projects|project experience)/.test(content),
    actionWords: /(developed|built|designed|implemented|led|created|optimized|managed)/.test(content),
    measurable: /(\d+%|\$\d+|\d+\+?\s*(users|clients|projects|requests|employees|months|years))/.test(content),
    readability: content.length >= 500 && content.length <= 12000,
  };
  const weights = { contact: 15, skills: 15, experience: 15, education: 10, projects: 10, actionWords: 10, measurable: 15, readability: 10 };
  const score = Math.max(0, Math.min(100, Math.round(Object.entries(checks).reduce((sum, [key, ok]) => sum + (ok ? weights[key] : 0), 0))));
  return {
    score,
    checks,
    matchedKeywords: [],
    missingKeywords: [],
    strengths: Object.keys(checks).filter((k) => checks[k]),
    improvements: Object.keys(checks).filter((k) => !checks[k]),
    targetRole: profile.targetRole || '',
    mode: 'generic',
  };
}

async function calculateATSWithLLM(text, targetRole, jobDescription) {
  const context = jobDescription?.trim()
    ? `Job description the candidate is applying against:\n"""${jobDescription.trim().slice(0, 4000)}"""`
    : targetRole?.trim()
      ? `Target role the candidate is applying for: ${targetRole.trim()}`
      : '';
  if (!context) return null;

  const systemPrompt =
    'You are an ATS (Applicant Tracking System) resume scoring engine, like the automated screeners used by recruiters. ' +
    'Score how well the resume matches the specific target role or job description - keyword coverage, relevant skills, ' +
    'relevant experience, and formatting/readability. Be strict and specific, not generic. Respond with a single JSON object and nothing else.';
  const userPrompt =
    `Resume text:\n"""${text.slice(0, 10000)}"""\n\n${context}\n\n` +
    'Return a JSON object with exactly these keys:\n' +
    '- score (integer 0-100: overall ATS match score for this specific role/JD)\n' +
    '- checks (object with boolean values for keys: contact, skills, experience, education, projects, actionWords, measurable, readability, keywordMatch, roleAlignment)\n' +
    '- matchedKeywords (array of important skills/keywords from the role/JD that ARE present in the resume)\n' +
    '- missingKeywords (array of important skills/keywords from the role/JD that are MISSING from the resume)\n' +
    '- strengths (array of short strings describing what makes this resume a good fit)\n' +
    '- improvements (array of short, actionable strings specific to closing the gap with this role/JD)';

  const parsed = await groqChatJSON({
    systemPrompt,
    userPrompt,
    temperature: 0.2,
    maxTokens: 900,
    label: 'ATS LLM scoring',
  });
  if (!parsed) return null;

  return {
    score: Math.max(0, Math.min(100, Math.round(Number(parsed.score) || 0))),
    checks: parsed.checks && typeof parsed.checks === 'object' ? parsed.checks : {},
    matchedKeywords: Array.isArray(parsed.matchedKeywords) ? parsed.matchedKeywords : [],
    missingKeywords: Array.isArray(parsed.missingKeywords) ? parsed.missingKeywords : [],
    strengths: Array.isArray(parsed.strengths) ? parsed.strengths : [],
    improvements: Array.isArray(parsed.improvements) ? parsed.improvements : [],
    targetRole: targetRole || '',
    mode: 'role-matched',
  };
}

async function calculateATS(text = '', profile = {}, options = {}) {
  const targetRole = (options.targetRole || profile.targetRole || '').trim();
  const jobDescription = (options.jobDescription || '').trim();
  const llmResult = await calculateATSWithLLM(text, targetRole, jobDescription);
  if (llmResult) return llmResult;
  return heuristicATS(text, profile);
}

module.exports = { calculateATS };
