import { useState, useEffect, useRef } from 'react';

export const useMediaStream = () => {
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);
  const [hasPermission, setHasPermission] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    let activeStream = null;

    const startStream = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({ 
          video: true, 
          audio: true 
        });
        setStream(mediaStream);
        setHasPermission(true);
        activeStream = mediaStream;

        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (err) {
        console.error('Error accessing media devices:', err);
        setError(err.message || 'Permission denied');
        setHasPermission(false);
      }
    };

    startStream();

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Sync stream to ref if ref changes
  useEffect(() => {
    if (videoRef.current && stream && videoRef.current.srcObject !== stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream, videoRef]);

  return { videoRef, stream, hasPermission, error };
};
