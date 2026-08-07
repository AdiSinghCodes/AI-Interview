import React, { useState, useEffect } from 'react';
import './WarningBanner.css';

const WarningBanner = ({ message, type = 'warning', count = 0 }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (message) {
      setIsVisible(true);
      const timer = setTimeout(() => {
        setIsVisible(false);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [message, count]); // re-trigger if count changes

  if (!message) return null;

  return (
    <div className={`warning-banner ${type} ${isVisible ? 'visible' : 'hidden'}`}>
      <div className="warning-icon">
        {type === 'warning' ? '⚠️' : '🚨'}
      </div>
      <div className="warning-content">
        <div className="warning-title">Proctoring Alert</div>
        <div className="warning-message">{message}</div>
      </div>
      {count > 0 && (
        <div className="warning-count">
          Warning {count}/3
        </div>
      )}
    </div>
  );
};

export default WarningBanner;
