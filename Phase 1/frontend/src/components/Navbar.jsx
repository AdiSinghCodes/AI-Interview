import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar = ({ isConnected }) => {
  const location = useLocation();
  const isInterviewPage = location.pathname === '/interview';

  return (
    <nav className="navbar glass-panel">
      <div className="navbar-brand">
        <Link to="/">
          <div className="logo">
            <span className="logo-icon">🤖</span>
            <span className="logo-text text-gradient">AI Interview Pro</span>
          </div>
        </Link>
      </div>

      <div className="navbar-links">
        {!isInterviewPage && (
          <>
            <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Home</Link>
            <Link to="/select" className={`nav-link ${location.pathname === '/select' ? 'active' : ''}`}>Practice</Link>
          </>
        )}
      </div>

      <div className="navbar-actions">
        {isInterviewPage && (
          <div className="connection-status">
            <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></div>
            <span className="status-text">{isConnected ? 'Connected' : 'Reconnecting...'}</span>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
