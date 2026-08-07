import React from 'react';
import { Link } from 'react-router-dom';
import './Landing.css';

const Landing = () => {
  return (
    <div className="landing-page">
      <div className="hero-section">
        <div className="hero-content animate-slide-up">
          <h1 className="hero-title text-gradient">
            Master Your Interview Skills with AI
          </h1>
          <p className="hero-subtitle">
            Practice with an AI-powered interviewer that provides real-time feedback on your answers, voice, body language, and confidence
          </p>
          <Link to="/select" className="btn-primary start-btn">
            Start Practice
          </Link>
        </div>
        
        <div className="stats-container animate-fade-in" style={{ animationDelay: '0.3s' }}>
          <div className="stat-item">
            <span className="stat-value">1000+</span>
            <span className="stat-label">Questions</span>
          </div>
          <div className="stat-divider"></div>
          <div className="stat-item">
            <span className="stat-value">6</span>
            <span className="stat-label">Domains</span>
          </div>
          <div className="stat-divider"></div>
          <div className="stat-item">
            <span className="stat-value">Real-time</span>
            <span className="stat-label">Analysis</span>
          </div>
        </div>
      </div>

      <div className="features-section">
        <div className="features-grid">
          <div className="feature-card glass-panel animate-slide-up" style={{ animationDelay: '0.4s' }}>
            <div className="feature-icon">🤖</div>
            <h3>AI Interviewer</h3>
            <p>Adaptive questions powered by advanced AI that tailor to your responses and role.</p>
          </div>
          
          <div className="feature-card glass-panel animate-slide-up" style={{ animationDelay: '0.5s' }}>
            <div className="feature-icon">🎙️</div>
            <h3>Voice Analysis</h3>
            <p>Real-time feedback on your speaking rate, volume, and use of filler words.</p>
          </div>
          
          <div className="feature-card glass-panel animate-slide-up" style={{ animationDelay: '0.6s' }}>
            <div className="feature-icon">👁️</div>
            <h3>Smart Proctoring</h3>
            <p>Monitor your body language, eye contact, and focus during the interview.</p>
          </div>
          
          <div className="feature-card glass-panel animate-slide-up" style={{ animationDelay: '0.7s' }}>
            <div className="feature-icon">📊</div>
            <h3>Instant Feedback</h3>
            <p>Detailed scores and actionable improvement tips immediately after your session.</p>
          </div>
        </div>
      </div>
      
      {/* Background elements */}
      <div className="bg-gradient-1"></div>
      <div className="bg-gradient-2"></div>
    </div>
  );
};

export default Landing;
