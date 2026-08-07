import React from 'react';
import './VoiceMeter.css';

const VoiceMeter = ({ level = 0, isSpeaking = false }) => {
  // Determine status based on volume
  let status = 'Silent';
  let statusClass = 'silent';
  
  if (isSpeaking) {
    if (level < 25) {
      status = 'Too Quiet';
      statusClass = 'quiet';
    } else if (level > 85) {
      status = 'Too Loud';
      statusClass = 'loud';
    } else {
      status = 'Good';
      statusClass = 'good';
    }
  }

  return (
    <div className="voice-meter-container">
      <div className="voice-meter-header">
        <span className="meter-label">Voice Level</span>
        <span className={`meter-status ${statusClass}`}>{status}</span>
      </div>
      
      <div className="meter-bar-bg">
        <div 
          className={`meter-bar-fill ${statusClass} ${isSpeaking ? 'active' : ''}`}
          style={{ width: `${Math.max(2, level)}%` }}
        ></div>
        
        {/* Markers */}
        <div className="meter-marker quiet-mark"></div>
        <div className="meter-marker loud-mark"></div>
      </div>
      
      {/* Waveform visual effect when speaking */}
      {isSpeaking && (
        <div className="waveform-mini">
          <div className="bar b1"></div>
          <div className="bar b2"></div>
          <div className="bar b3"></div>
          <div className="bar b4"></div>
          <div className="bar b5"></div>
        </div>
      )}
    </div>
  );
};

export default VoiceMeter;
