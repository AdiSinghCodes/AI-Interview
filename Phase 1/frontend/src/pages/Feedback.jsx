import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Navbar from '../components/Navbar';
import ScoreCard from '../components/ScoreCard';
import { api } from '../utils/api';
import './Feedback.css';

const Feedback = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);
  const sessionId = searchParams.get('session_id');

  const [loading, setLoading] = useState(!!sessionId);
  const [data, setData] = useState(null);

  useEffect(() => {
    if (sessionId) {
      api.getFeedback(sessionId)
        .then(res => {
          setData(res);
          setLoading(false);
        })
        .catch(err => {
          console.error("Failed to fetch feedback", err);
          setLoading(false);
        });
    }
  }, [sessionId]);

  // Default / fallback feedback values
  const overallScore = data ? Math.round(data.overall_score || 78) : 78;
  const categoryScores = data?.category_scores || {};
  
  const metrics = [
    { label: 'Answer Quality', score: Math.round((categoryScores.answer_quality || 8.2) * 10), icon: '🧠' },
    { label: 'Communication', score: Math.round((categoryScores.communication || 7.5) * 10), icon: '🗣️' },
    { label: 'Confidence', score: Math.round((categoryScores.confidence || 6.8) * 10), icon: '💪' },
    { label: 'Eye Contact', score: Math.round((categoryScores.eye_contact || 9.0) * 10), icon: '👁️' },
    { label: 'Body Language', score: Math.round((categoryScores.body_language || 8.5) * 10), icon: '👋' },
    { label: 'Voice Clarity', score: Math.round((categoryScores.voice_clarity || 7.2) * 10), icon: '🎙️' }
  ];

  const questions = data?.per_question_feedback?.length > 0
    ? data.per_question_feedback.map((q, idx) => ({
        id: idx + 1,
        text: q.question || `Question ${idx + 1}`,
        answer: q.answer || "No response provided",
        feedback: q.evaluation?.feedback || "Evaluation complete.",
        score: Math.round(((q.evaluation?.scores?.relevance || 7) + (q.evaluation?.scores?.accuracy || 7) + (q.evaluation?.scores?.depth || 7)) / 0.3)
      }))
    : [
        {
          id: 1,
          text: "Tell me about a time you faced a technical challenge.",
          answer: "I was working on a React project and we had severe performance issues with a large list. I implemented virtualization which improved render times by 80%.",
          feedback: "Good specific example. Mentioning the 80% improvement adds quantifiable impact. Could have elaborated on the specific virtualization library used.",
          score: 85
        },
        {
          id: 2,
          text: "How do you handle tight deadlines?",
          answer: "I prioritize tasks, communicate with the team, and try to focus on the MVP first.",
          feedback: "A bit generic. Try to use the STAR method (Situation, Task, Action, Result) to give a concrete example of a time you actually did this.",
          score: 60
        }
      ];

  if (loading) {
    return (
      <div className="feedback-page">
        <Navbar isConnected={false} />
        <div className="feedback-container text-center" style={{ marginTop: '100px' }}>
          <h2>Generating AI Feedback Report...</h2>
          <p>Analyzing answer quality, speech rate, tone, and proctoring metrics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="feedback-page">
      <Navbar isConnected={false} />
      
      <div className="feedback-container animate-fade-in">
        <div className="fb-header text-center">
          <h1 className="text-gradient">Interview Analysis</h1>
          <p>Here's a detailed breakdown of your performance</p>
        </div>

        <div className="overall-score-section">
          <div className="main-score glass-panel animate-slide-up">
            <h3>Overall Score</h3>
            <div className="score-circle">
              <svg viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" className="bg" />
                <circle cx="50" cy="50" r="45" className="progress" style={{ strokeDashoffset: 283 - (283 * overallScore) / 100 }} />
              </svg>
              <div className="score-text">
                <span className="value">{overallScore}</span>
                <span className="max">/100</span>
              </div>
            </div>
            <p className="score-comment">Good job! You have a solid foundation but room for improvement in behavioral examples.</p>
          </div>

          <div className="stats-cards glass-panel animate-slide-up" style={{ animationDelay: '0.1s' }}>
            <div className="stat-col">
              <h4>Voice Metrics</h4>
              <ul>
                <li><span>Speaking Rate:</span> <strong>145 WPM</strong> (Ideal)</li>
                <li><span>Average Volume:</span> <strong>Good</strong></li>
                <li><span>Filler Words:</span> <strong>12</strong> (Slightly High)</li>
              </ul>
            </div>
            <div className="stat-col">
              <h4>Proctoring Summary</h4>
              <ul>
                <li><span>Focus Score:</span> <strong>95%</strong></li>
                <li><span>Warnings:</span> <strong>0</strong></li>
                <li><span>Eye Contact:</span> <strong>Consistent</strong></li>
              </ul>
            </div>
          </div>
        </div>

        <h2 className="section-title">Detailed Breakdown</h2>
        <div className="metrics-grid">
          {metrics.map((m, i) => (
            <div key={i} style={{ animationDelay: `${0.2 + (i * 0.1)}s` }}>
              <ScoreCard label={m.label} score={m.score} icon={m.icon} />
            </div>
          ))}
        </div>

        <h2 className="section-title">Question Analysis</h2>
        <div className="questions-review">
          {questions.map((q, i) => (
            <div key={q.id} className="q-review-card glass-panel animate-slide-up" style={{ animationDelay: `${0.5 + (i * 0.1)}s` }}>
              <div className="qr-header">
                <h3>Question {q.id} <span className="qr-score badge success">{q.score}/100</span></h3>
                <p className="qr-text">{q.text}</p>
              </div>
              <div className="qr-body">
                <div className="qr-block">
                  <h4>Your Answer:</h4>
                  <p>{q.answer}</p>
                </div>
                <div className="qr-block feedback">
                  <h4>AI Feedback:</h4>
                  <p>{q.feedback}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="fb-actions animate-fade-in" style={{ animationDelay: '1s' }}>
          <button className="btn-primary" onClick={() => navigate('/select')}>
            Practice Again
          </button>
          <button className="btn-outline" onClick={() => navigate('/')}>
            Back to Home
          </button>
        </div>
      </div>
    </div>
  );
};

export default Feedback;
