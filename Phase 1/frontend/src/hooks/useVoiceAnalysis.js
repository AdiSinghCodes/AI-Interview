import { useState, useEffect, useRef } from 'react';

export const useVoiceAnalysis = (audioStream) => {
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [averageVolume, setAverageVolume] = useState(0);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const rafRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    if (!audioStream) return;

    const initAudio = () => {
      try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(audioStream);
        const analyser = audioContext.createAnalyser();
        
        analyser.fftSize = 256;
        source.connect(analyser);
        
        audioContextRef.current = audioContext;
        analyserRef.current = analyser;

        let volumeSum = 0;
        let count = 0;

        const updateVolume = () => {
          const dataArray = new Uint8Array(analyser.frequencyBinCount);
          analyser.getByteFrequencyData(dataArray);
          
          let sum = 0;
          for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
          }
          
          const rms = Math.sqrt(sum / dataArray.length);
          // Map to 0-100 roughly
          const level = Math.min(100, Math.round(rms * 1.5));
          
          setVolumeLevel(level);
          setIsSpeaking(level > 15); // Threshold for speaking
          
          volumeSum += level;
          count++;
          if (count % 30 === 0) { // Update average periodically
            setAverageVolume(Math.round(volumeSum / count));
          }

          rafRef.current = requestAnimationFrame(updateVolume);
        };

        updateVolume();
      } catch (err) {
        console.error('Error initializing audio context:', err);
      }
    };

    initAudio();

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, [audioStream]);

  const startRecording = (onDataAvailable) => {
    if (!audioStream || isRecording) return;
    
    try {
      const mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && onDataAvailable) {
          onDataAvailable(event.data);
        }
      };
      
      mediaRecorder.start(1000); // chunk every 1 second
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
    } catch (err) {
      console.error('Error starting recording:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return { 
    volumeLevel, 
    isSpeaking, 
    averageVolume, 
    startRecording, 
    stopRecording, 
    isRecording 
  };
};
