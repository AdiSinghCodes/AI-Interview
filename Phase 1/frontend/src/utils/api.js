import { API_BASE_URL } from './constants';

const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'An unknown error occurred' }));
    throw new Error(error.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
};

export const api = {
  get: (endpoint) => fetch(`${API_BASE_URL}${endpoint}`).then(handleResponse),
  
  post: (endpoint, data) => fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  }).then(handleResponse),
  
  uploadAudio: async (endpoint, audioBlob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(response);
  },

  // Task 1: New Backend Calls
  startInterview: (role, difficulty, numQuestions) => 
    api.post('/interview/start', { role, difficulty, num_questions: numQuestions }),
    
  submitAnswer: (sessionId, answerText, voiceMetrics) => 
    api.post(`/interview/${sessionId}/answer`, { answer_text: answerText, voice_metrics: voiceMetrics || {} }),
    
  getQuestion: (sessionId) => 
    api.get(`/interview/${sessionId}/question`),
    
  endInterview: (sessionId) => 
    api.post(`/interview/${sessionId}/end`, {}),
    
  getTTSAudio: async (text) => {
    const response = await fetch(`${API_BASE_URL}/tts/speak?text=${encodeURIComponent(text)}`);
    if (!response.ok) throw new Error('TTS failed');
    return await response.blob();
  },
  
  getFeedback: (sessionId) => 
    api.get(`/feedback/${sessionId}`)
};
