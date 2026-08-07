import React from 'react';
import './ScoreCard.css';

const ScoreCard = ({ label, score = 0, icon }) => {
  // Determine color based on score
  let statusClass = 'high';
  if (score < 40) statusClass = 'low';
  else if (score < 70) statusClass = 'medium';

  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="score-card glass-panel animate-slide-up">
      <div className="sc-header">
        <span className="sc-icon">{icon}</span>
        <span className="sc-label">{label}</span>
      </div>
      
      <div className="sc-body">
        <div className="sc-circular-progress">
          <svg width="80" height="80" viewBox="0 0 80 80">
            <circle 
              className="sc-circle-bg" 
              cx="40" cy="40" r={radius} 
              fill="none" strokeWidth="6" 
            />
            <circle 
              className={`sc-circle-fill ${statusClass}`} 
              cx="40" cy="40" r={radius} 
              fill="none" strokeWidth="6"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              transform="rotate(-90 40 40)"
            />
          </svg>
          <div className="sc-score-value">
            {Math.round(score)}<span className="sc-percent">%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScoreCard;
