import React from 'react';
import './QuestionCard.css';

const QuestionCard = ({ 
  questionText = "Could you tell me about a time you faced a significant technical challenge and how you overcame it?",
  number = 1,
  total = 5,
  category = "Behavioral",
  difficulty = "medium"
}) => {
  return (
    <div className="question-card glass-panel animate-slide-up">
      <div className="qc-header">
        <div className="qc-meta">
          <span className="qc-number">Question {number} of {total}</span>
          <span className="qc-category">{category}</span>
        </div>
        <div className="qc-difficulty">
          <span className={`diff-dot ${difficulty === 'easy' ? 'active' : ''}`}></span>
          <span className={`diff-dot ${difficulty === 'medium' || difficulty === 'hard' ? 'active' : ''}`}></span>
          <span className={`diff-dot ${difficulty === 'hard' ? 'active' : ''}`}></span>
        </div>
      </div>
      <div className="qc-body">
        <h3 className="qc-text">{questionText}</h3>
      </div>
    </div>
  );
};

export default QuestionCard;
