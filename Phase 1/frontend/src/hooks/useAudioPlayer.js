import { useState, useEffect, useRef, useCallback } from 'react';

export const useAudioPlayer = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [amplitude, setAmplitude] = useState(0);
  
  const audioRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const animationFrameRef = useRef(null);

  const initAudio = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
    }
  }, []);

  const updateAmplitude = useCallback(() => {
    if (!analyserRef.current || !isPlaying) return;
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      sum += dataArray[i];
    }
    const avg = sum / dataArray.length;
    const normalizedAmplitude = Math.min(1, avg / 128); // 0 to 1
    
    setAmplitude(normalizedAmplitude);
    
    animationFrameRef.current = requestAnimationFrame(updateAmplitude);
  }, [isPlaying]);

  useEffect(() => {
    if (isPlaying) {
      animationFrameRef.current = requestAnimationFrame(updateAmplitude);
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      setAmplitude(0);
    }
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, updateAmplitude]);

  const play = useCallback(async (audioSource) => {
    initAudio();
    
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }

    if (audioRef.current) {
      audioRef.current.pause();
    }

    let url = audioSource;
    if (audioSource instanceof Blob) {
      url = URL.createObjectURL(audioSource);
    }

    audioRef.current = new Audio(url);
    
    if (sourceRef.current) {
        sourceRef.current.disconnect();
    }
    
    try {
      sourceRef.current = audioContextRef.current.createMediaElementSource(audioRef.current);
      sourceRef.current.connect(analyserRef.current);
      analyserRef.current.connect(audioContextRef.current.destination);
    } catch (e) {
      console.error("Audio routing error:", e);
    }

    audioRef.current.onended = () => {
      setIsPlaying(false);
      setAmplitude(0);
    };

    audioRef.current.onplay = () => {
      setIsPlaying(true);
    };

    audioRef.current.onpause = () => {
      setIsPlaying(false);
      setAmplitude(0);
    };

    try {
      await audioRef.current.play();
    } catch (e) {
      console.error("Playback failed:", e);
      setIsPlaying(false);
    }
  }, [initAudio]);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsPlaying(false);
    setAmplitude(0);
  }, []);

  useEffect(() => {
    return () => {
      stop();
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(console.error);
      }
    };
  }, [stop]);

  return { play, stop, isPlaying, amplitude };
};
