import { useState, useEffect, useCallback } from 'react';

export const useProctoring = (isActive = true) => {
  const [violations, setViolations] = useState([]);
  const [warningCount, setWarningCount] = useState(0);

  const addViolation = useCallback((type, message) => {
    if (!isActive) return;
    
    const violation = {
      type,
      message,
      timestamp: new Date().toISOString()
    };
    
    setViolations(prev => [...prev, violation]);
    setWarningCount(prev => prev + 1);
  }, [isActive]);

  useEffect(() => {
    if (!isActive) return;

    // Visibility change (tab switch)
    const handleVisibilityChange = () => {
      if (document.hidden) {
        addViolation('tab_switch', 'Switched away from interview tab');
      }
    };

    // Window blur (lost focus)
    const handleBlur = () => {
      addViolation('window_blur', 'Window lost focus');
    };

    // Prevent copy/paste
    const handleCopyPaste = (e) => {
      e.preventDefault();
      addViolation('copy_paste', 'Copy/paste operations are not allowed');
    };

    // Prevent context menu (right click)
    const handleContextMenu = (e) => {
      e.preventDefault();
      addViolation('right_click', 'Right-click is disabled during the interview');
    };

    // Fullscreen exit
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement) {
        addViolation('fullscreen_exit', 'Exited fullscreen mode');
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('blur', handleBlur);
    document.addEventListener('copy', handleCopyPaste);
    document.addEventListener('paste', handleCopyPaste);
    document.addEventListener('contextmenu', handleContextMenu);
    document.addEventListener('fullscreenchange', handleFullscreenChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('blur', handleBlur);
      document.removeEventListener('copy', handleCopyPaste);
      document.removeEventListener('paste', handleCopyPaste);
      document.removeEventListener('contextmenu', handleContextMenu);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [isActive, addViolation]);

  return { violations, warningCount, addViolation };
};
