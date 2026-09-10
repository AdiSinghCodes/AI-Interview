function calculateATS(text = '', profile = {}) {
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
  return { score, checks, strengths: Object.keys(checks).filter(k => checks[k]), improvements: Object.keys(checks).filter(k => !checks[k]) };
}
module.exports = { calculateATS };
