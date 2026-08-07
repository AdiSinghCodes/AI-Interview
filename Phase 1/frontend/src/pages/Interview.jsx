import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Avatar from '../components/Avatar';
import QuestionCard from '../components/QuestionCard';
import Webcam from '../components/Webcam';
import VoiceMeter from '../components/VoiceMeter';
import Timer from '../components/Timer';
import WarningBanner from '../components/WarningBanner';
import { useMediaStream } from '../hooks/useMediaStream';
import { useVoiceAnalysis } from '../hooks/useVoiceAnalysis';
import { useProctoring } from '../hooks/useProctoring';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAudioPlayer } from '../hooks/useAudioPlayer';
import { WS_BASE_URL } from '../utils/constants';
import { api } from '../utils/api';
import './Interview.css';

const Interview = () => {
  const navigate = useNavigate();
  const [interviewState, setInterviewState] = useState('connecting'); // connecting, greeting, asking, listening, evaluating, complete
  const [currentQuestion, setCurrentQuestion] = useState(1);
  const [totalQuestions, setTotalQuestions] = useState(5);
  const [questionText, setQuestionText] = useState("Connecting to AI Interviewer...");
  const [sessionId, setSessionId] = useState(null);
  const [transcript, setTranscript] = useState('');
  
  // Hooks
  const { videoRef, stream, hasPermission, error } = useMediaStream();
  const { volumeLevel, isSpeaking: userSpeaking } = useVoiceAnalysis(stream);
  const { violations, warningCount, addViolation } = useProctoring(interviewState !== 'connecting' && interviewState !== 'complete');
  const { isConnected, lastMessage, sendMessage } = useWebSocket(sessionId ? `${WS_BASE_URL}/interview/${sessionId}` : null);
  const { play, stop, isPlaying: aiAudioPlaying, amplitude: aiAmplitude } = useAudioPlayer();

  const recognitionRef = useRef(null);

  // Send webcam frames to backend WebSocket for AI Proctoring
  useEffect(() => {
    if (!sessionId || !isConnected) return;
    const interval = setInterval(() => {
      if (videoRef.current && videoRef.current.readyState === 4) {
        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 240;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, 320, 240);
        const frame = canvas.toDataURL('image/jpeg', 0.5);
        sendMessage({ type: 'video_frame', frame });
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [sessionId, isConnected, sendMessage, videoRef]);

  // Handle AI Proctoring alerts from WebSocket
  useEffect(() => {
    if (lastMessage?.type === 'proctoring_update' && lastMessage?.data?.messages?.length > 0) {
      lastMessage.data.messages.forEach(msg => {
        addViolation('ai_proctoring', msg);
      });
    }
  }, [lastMessage, addViolation]);

  // Initialize SpeechRecognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      
      recognition.onresult = (event) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          currentTranscript += event.results[i][0].transcript;
        }
        setTranscript(prev => {
           return prev + ' ' + currentTranscript;
        });
      };
      
      recognition.onerror = (event) => {
        console.error("Speech recognition error", event.error);
      };
      
      recognitionRef.current = recognition;
    } else {
      console.warn("Speech Recognition not supported in this browser.");
    }
  }, []);

  useEffect(() => {
    const startInterviewProcess = async () => {
      try {
        const rawSettings = JSON.parse(localStorage.getItem('interview_settings')) || {};
        const role = rawSettings.role || rawSettings.roleId || 'React Developer';
        const difficulty = rawSettings.difficulty || 'medium';
        const numQuestions = rawSettings.num_questions || rawSettings.questionCount || 5;
        
        const res = await api.startInterview(role, difficulty, numQuestions);
        setSessionId(res.session_id);
        setTotalQuestions(numQuestions);
        setCurrentQuestion(1);
        
        const qObj = res.first_question;
        const qStr = typeof qObj === 'object' ? qObj?.question || '' : qObj;
        const fullText = `${res.greeting} ${qStr}`;
        setQuestionText(qStr);
        
        setInterviewState('asking');
        try {
          const audioBlob = await api.getTTSAudio(fullText);
          await play(audioBlob);
        } catch(e) {
          console.error("Failed to play TTS audio", e);
          setInterviewState('listening');
        }
      } catch (err) {
        console.error("Error starting interview:", err);
        setQuestionText("Failed to start interview. Please try again.");
      }
    };

    if (interviewState === 'connecting') {
      startInterviewProcess();
    }
  }, [interviewState, play]);

  // Audio end transitions
  useEffect(() => {
    if (!aiAudioPlaying && interviewState === 'asking') {
      setInterviewState('listening');
    }
  }, [aiAudioPlaying, interviewState]);

  // Manage listening state
  useEffect(() => {
    if (interviewState === 'listening') {
       if (recognitionRef.current) {
          setTranscript('');
          try {
             recognitionRef.current.start();
          } catch(e) {
             // Already started or active
          }
       }
    } else {
       if (recognitionRef.current) {
          try {
             recognitionRef.current.stop();
          } catch(e) {}
       }
    }
  }, [interviewState]);

  const handleNextState = async () => {
    if (interviewState === 'listening') {
      setInterviewState('evaluating');
      if (recognitionRef.current) {
         recognitionRef.current.stop();
      }
      
      try {
        // Just send transcript for answer text
        const res = await api.submitAnswer(sessionId, transcript, { volume: volumeLevel });
        
        if (currentQuestion < totalQuestions && res.next_question) {
          const nextQObj = res.next_question;
          const nextQStr = typeof nextQObj === 'object' ? nextQObj?.question || '' : nextQObj;
          setCurrentQuestion(c => c + 1);
          setQuestionText(nextQStr);
          setInterviewState('asking');
          try {
            const audioBlob = await api.getTTSAudio(nextQStr);
            await play(audioBlob);
          } catch(e) {
            console.error(e);
            setInterviewState('listening');
          }
        } else {
          setInterviewState('complete');
          api.endInterview(sessionId).catch(console.error);
          navigate(`/feedback?session_id=${sessionId}`);
        }
      } catch (err) {
        console.error("Error submitting answer:", err);
        setInterviewState('complete');
        navigate(`/feedback?session_id=${sessionId}`);
      }
    }
  };

  const handleEndInterview = () => {
    if (sessionId) {
       api.endInterview(sessionId).catch(console.error);
       navigate(`/feedback?session_id=${sessionId}`);
    } else {
       navigate('/feedback');
    }
  };

  const isAiSpeaking = aiAudioPlaying;
  const latestViolation = violations.length > 0 ? violations[violations.length - 1].message : '';

  return (
    <div className="interview-page">
      <Navbar isConnected={isConnected} />
      
      <WarningBanner message={latestViolation} type="danger" count={warningCount} />

      <div className="interview-content">
        {/* Left Column - AI Interaction */}
        <div className="split-left">
          <div className="avatar-section glass-panel">
            <Avatar 
              isSpeaking={isAiSpeaking} 
              emotion={interviewState === 'evaluating' ? 'thinking' : 'neutral'}
              amplitude={aiAmplitude}
            />
            <div className={`ai-status-badge ${interviewState}`}>
              {interviewState === 'connecting' && 'Connecting...'}
              {interviewState === 'greeting' && 'Greeting'}
              {interviewState === 'asking' && 'AI is speaking...'}
              {interviewState === 'listening' && 'AI is listening...'}
              {interviewState === 'evaluating' && 'Evaluating answer...'}
              {interviewState === 'complete' && 'Interview Complete'}
            </div>
            {isAiSpeaking && (
              <div className="waveform-container">
                <div className="waveform-bar" style={{ height: `${20 + aiAmplitude * 30}px` }}></div>
                <div className="waveform-bar" style={{ height: `${10 + aiAmplitude * 50}px` }}></div>
                <div className="waveform-bar" style={{ height: `${20 + aiAmplitude * 30}px` }}></div>
              </div>
            )}
          </div>
          
          <div className="question-section">
            <QuestionCard 
              questionText={questionText}
              number={currentQuestion}
              total={totalQuestions}
              category="Behavioral"
              difficulty="medium"
            />
          </div>
        </div>

        {/* Right Column - User Feed & Stats */}
        <div className="split-right">
          <div className="webcam-section webcam-active-border">
            <Webcam 
              videoRef={videoRef}
              hasPermission={hasPermission}
              error={error}
            />
          </div>
          
          <div className="voice-meter-section">
            <VoiceMeter level={volumeLevel} isSpeaking={userSpeaking} />
          </div>
          
          <div className="proctoring-stats glass-panel">
            <h3>Proctoring Status</h3>
            <div className="stat-row">
              <span>Violations</span>
              <span className={`badge ${warningCount > 0 ? 'warning' : 'success'}`}>{warningCount}</span>
            </div>
            <div className="stat-row">
              <span>Eye Contact</span>
              <span className="badge success">Good</span>
            </div>
            {interviewState === 'listening' && (
              <div className="stat-row">
                <span>Transcript length</span>
                <span className="badge">{transcript.length}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Control Bar */}
      <div className="bottom-bar glass-panel">
        <div className="bb-left">
          <Timer isRunning={interviewState === 'listening'} />
          {interviewState === 'listening' && (
            <div className="recording-indicator animate-pulse">
              <span className="rec-dot pulsing-dot"></span> Recording Answer
            </div>
          )}
        </div>
        
        <div className="bb-center">
          {interviewState === 'listening' && (
            <button className="btn-primary status-transition" onClick={handleNextState}>
              Finish Answer
            </button>
          )}
        </div>
        
        <div className="bb-right">
          <button className="btn-outline danger" onClick={handleEndInterview}>
            End Interview
          </button>
        </div>
      </div>
    </div>
  );
};

export default Interview;

