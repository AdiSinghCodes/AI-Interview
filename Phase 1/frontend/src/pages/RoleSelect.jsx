import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ROLES, DIFFICULTY_LEVELS, QUESTION_COUNTS, DEFAULT_SETTINGS } from '../utils/constants';
import './RoleSelect.css';

const RoleSelect = () => {
  const navigate = useNavigate();
  const [selectedRole, setSelectedRole] = useState(DEFAULT_SETTINGS.roleId);
  const [difficulty, setDifficulty] = useState(DEFAULT_SETTINGS.difficulty);
  const [questionCount, setQuestionCount] = useState(DEFAULT_SETTINGS.questionCount);

  const handleStart = () => {
    if (!selectedRole) return;
    
    // Save settings to localStorage or context
    localStorage.setItem('interview_settings', JSON.stringify({
      roleId: selectedRole,
      difficulty,
      questionCount
    }));
    
    navigate('/interview');
  };

  return (
    <div className="role-select-page animate-fade-in">
      <div className="rs-container">
        <h1 className="rs-title text-gradient">Choose Your Interview Domain</h1>
        <p className="rs-subtitle">Select a role and configure your practice session.</p>

        <div className="roles-grid">
          {ROLES.map((role) => (
            <div 
              key={role.id}
              className={`role-card glass-panel ${selectedRole === role.id ? 'selected' : ''}`}
              onClick={() => setSelectedRole(role.id)}
            >
              <div className="role-icon">{role.icon}</div>
              <h3 className="role-title">{role.title}</h3>
              <p className="role-desc">{role.description}</p>
              
              {selectedRole === role.id && (
                <div className="role-selected-indicator">
                  ✓
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="rs-config glass-panel">
          <div className="config-group">
            <h3>Difficulty Level</h3>
            <div className="pill-group">
              {DIFFICULTY_LEVELS.map(level => (
                <button
                  key={level.id}
                  className={`pill-btn ${difficulty === level.id ? 'active' : ''}`}
                  onClick={() => setDifficulty(level.id)}
                >
                  {level.label}
                </button>
              ))}
            </div>
          </div>

          <div className="config-group">
            <h3>Number of Questions</h3>
            <div className="pill-group">
              {QUESTION_COUNTS.map(count => (
                <button
                  key={count.id}
                  className={`pill-btn ${questionCount === count.id ? 'active' : ''}`}
                  onClick={() => setQuestionCount(count.id)}
                >
                  {count.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="rs-action">
          <button 
            className="btn-primary begin-btn" 
            onClick={handleStart}
            disabled={!selectedRole}
          >
            Begin Interview
          </button>
        </div>
      </div>
    </div>
  );
};

export default RoleSelect;
