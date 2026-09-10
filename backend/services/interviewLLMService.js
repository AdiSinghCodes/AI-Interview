const axios = require('axios');

const AGENT_URL = process.env.INTERVIEW_AGENT_URL || 'http://127.0.0.1:5100';

async function agent(path, payload, timeout = 90000) {
  const response = await axios.post(`${AGENT_URL}${path}`, payload, { timeout });
  return response.data;
}

exports.nextQuestion = (payload) => agent('/question', payload);
exports.evaluateAnswer = (payload) => agent('/evaluate', payload);
exports.transcribe = async (buffer, filename, mimeType) => {
  const FormData = require('form-data');
  const form = new FormData();
  form.append('audio', buffer, { filename, contentType: mimeType || 'audio/webm' });
  const response = await axios.post(`${AGENT_URL}/transcribe`, form, {
    headers: form.getHeaders(),
    timeout: 120000,
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
  });
  return response.data;
};
