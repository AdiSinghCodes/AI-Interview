export const API_BASE_URL = 'http://localhost:8000/api';
export const WS_BASE_URL = 'ws://localhost:8000/ws';

export const ROLES = [
  { id: 'hr', title: 'HR / Behavioral', icon: '👥', description: 'General behavioral and cultural fit questions' },
  { id: 'technical', title: 'General Technical', icon: '💻', description: 'Core computer science and problem solving' },
  { id: 'react', title: 'React Developer', icon: '⚛️', description: 'React, hooks, state management, and frontend architecture' },
  { id: 'python', title: 'Python Developer', icon: '🐍', description: 'Python syntax, OOP, data structures, and backend' },
  { id: 'data', title: 'Data Analyst', icon: '📊', description: 'SQL, statistics, data visualization, and analytical thinking' },
  { id: 'custom', title: 'Custom Role', icon: '✨', description: 'Tailor the interview to a specific job description' }
];

export const DIFFICULTY_LEVELS = [
  { id: 'easy', label: 'Easy' },
  { id: 'medium', label: 'Medium' },
  { id: 'hard', label: 'Hard' }
];

export const QUESTION_COUNTS = [
  { id: 5, label: '5 Questions' },
  { id: 10, label: '10 Questions' },
  { id: 15, label: '15 Questions' }
];

export const DEFAULT_SETTINGS = {
  roleId: 'react',
  difficulty: 'medium',
  questionCount: 5
};
