import React, { useState, useEffect } from 'react';
import './Timer.css';

const Timer = ({ initialSeconds = 0, isCountdown = false, onComplete, isRunning = true }) => {
  const [seconds, setSeconds] = useState(initialSeconds);
  
  // Calculate max time for countdown (assume 120s max for circle progress)
  const maxTime = initialSeconds > 0 ? initialSeconds : 120;
  
  useEffect(() => {
    let interval = null;
    
    if (isRunning) {
      interval = setInterval(() => {
        setSeconds(prev => {
          if (isCountdown) {
            if (prev <= 1) {
              clearInterval(interval);
              if (onComplete) onComplete();
              return 0;
            }
            return prev - 1;
          } else {
            return prev + 1;
          }
        });
      }, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRunning, isCountdown, onComplete]);

  // Format MM:SS
  const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
  const secs = (seconds % 60).toString().padStart(2, '0');
  
  // SVG Circle calculation
  const radius = 20;
  const circumference = 2 * Math.PI * radius;
  const progress = isCountdown 
    ? (seconds / maxTime) * circumference 
    : Math.min((seconds / maxTime) * circumference, circumference);

  const isWarning = isCountdown && seconds <= 15;

  return (
    <div className="timer-container">
      <svg className="timer-svg" width="50" height="50" viewBox="0 0 50 50">
        <circle 
          className="timer-bg" 
          cx="25" cy="25" r={radius} 
          fill="none" strokeWidth="4" 
        />
        <circle 
          className={`timer-progress ${isWarning ? 'warning' : ''}`} 
          cx="25" cy="25" r={radius} 
          fill="none" strokeWidth="4"
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
          transform="rotate(-90 25 25)"
        />
      </svg>
      <div className={`timer-text ${isWarning ? 'warning' : ''}`}>
        {mins}:{secs}
      </div>
    </div>
  );
};

export default Timer;
