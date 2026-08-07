import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_BASE_URL } from '../utils/constants';

export const useWebSocket = (url = WS_BASE_URL) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const connect = useCallback(() => {
    if (!url) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      wsRef.current = new WebSocket(url);
    } catch (err) {
      console.error("Failed to initialize WebSocket:", err);
      return;
    }

    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };

    wsRef.current.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected, attempting reconnect...');
      // Auto-reconnect
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      wsRef.current.close();
    };

    wsRef.current.onmessage = (event) => {
      try {
        if (typeof event.data === 'string') {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } else {
          // Handle binary data (audio)
          setLastMessage({ type: 'audio', data: event.data });
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const sendMessage = useCallback((msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      if (typeof msg === 'object' && !(msg instanceof Blob) && !(msg instanceof ArrayBuffer)) {
        wsRef.current.send(JSON.stringify(msg));
      } else {
        wsRef.current.send(msg);
      }
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  return { isConnected, lastMessage, sendMessage };
};
