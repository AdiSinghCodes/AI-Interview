const axios = require('axios');

const GROQ_API_KEY = (process.env.GROQ_API_KEY || '').trim();
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

const DEFAULT_MODELS = [
  process.env.RESUME_LLM_MODEL || process.env.INTERVIEW_LLM_MODEL || 'openai/gpt-oss-120b',
  'qwen/qwen3.8-27b',
  'qwen/qwen3.6-27b',
];

function extractJSON(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (_) {}
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    try {
      return JSON.parse(fenced[1]);
    } catch (_) {}
  }
  const braceMatch = text.match(/\{[\s\S]*\}/);
  if (braceMatch) {
    try {
      return JSON.parse(braceMatch[0]);
    } catch (_) {}
  }
  return null;
}

async function groqChatJSON({ systemPrompt, userPrompt, temperature = 0.2, maxTokens = 900, models = DEFAULT_MODELS, label = 'Groq' }) {
  if (!GROQ_API_KEY) return null;
  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];

  for (const model of models) {
    // Try with forced JSON mode first, then fall back to plain text + manual
    // JSON extraction for models (e.g. gpt-oss reasoning models) that Groq's
    // structured-output validator rejects before generation even completes.
    for (const jsonMode of [true, false]) {
      try {
        const payload = { model, messages, temperature, max_tokens: maxTokens };
        if (jsonMode) payload.response_format = { type: 'json_object' };
        const { data } = await axios.post(GROQ_URL, payload, {
          headers: { Authorization: `Bearer ${GROQ_API_KEY}`, 'Content-Type': 'application/json' },
          timeout: 25000,
        });
        const content = data?.choices?.[0]?.message?.content;
        const parsed = extractJSON(content);
        if (parsed) return parsed;
      } catch (err) {
        console.error(`${label} failed (model=${model}, jsonMode=${jsonMode}):`, err.response?.data || err.message);
      }
    }
  }
  return null;
}

module.exports = { groqChatJSON, GROQ_API_KEY };
