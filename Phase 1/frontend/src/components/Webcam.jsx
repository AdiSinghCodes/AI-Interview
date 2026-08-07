import React from 'react';
import './Webcam.css';

const Webcam = ({ videoRef, hasPermission, error }) => {
  return (
    <div className="webcam-container glass-panel">
      <div className="webcam-header">
        <div className="webcam-title">
          <span className={`webcam-status-dot ${hasPermission ? 'active' : 'inactive'}`}></span>
          Your Camera
        </div>
      </div>
      
      <div className="webcam-wrapper">
        {hasPermission ? (
          <>
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className="webcam-video"
            />
            <div className="webcam-overlay-vignette"></div>
            
            {/* Proctoring UI Overlays */}
            <div className="proctoring-corners">
              <div className="corner tl"></div>
              <div className="corner tr"></div>
              <div className="corner bl"></div>
              <div className="corner br"></div>
            </div>
            
            <div className="face-guide">
              <div className="face-box"></div>
            </div>
          </>
        ) : (
          <div className="webcam-fallback">
            <div className="camera-icon">📷</div>
            <p>{error || 'Waiting for camera permission...'}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Webcam;
