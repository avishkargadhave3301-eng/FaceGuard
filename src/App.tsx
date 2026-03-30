/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import { Camera, ShieldCheck, ShieldAlert, UserPlus, Trash2, Volume2, VolumeX, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// Configuration: Using a reliable CDN for the face-api models
const MODEL_URL = 'https://vladmandic.github.io/face-api/model/';
const DETECTION_INTERVAL = 600; // Slightly slower for higher accuracy processing
const DISTANCE_THRESHOLD = 0.5; // Stricter threshold (0.4-0.5 is high accuracy, 0.6 is balanced)

interface RegisteredFace {
  name: string;
  descriptor: Float32Array;
  image: string;
}

export default function App() {
  // Refs for DOM elements
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Application State
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'authorized' | 'unknown'>('loading');
  const [registeredFaces, setRegisteredFaces] = useState<RegisteredFace[]>([]);
  const registeredFacesRef = useRef<RegisteredFace[]>([]);

  // Update ref whenever state changes to avoid stale closures in setInterval
  useEffect(() => {
    registeredFacesRef.current = registeredFaces;
  }, [registeredFaces]);
  const [isAlarmEnabled, setIsAlarmEnabled] = useState(true);
  const [detectedName, setDetectedName] = useState<string>('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [newName, setNewName] = useState('');

  // Audio object for the alarm sound
  const alarmAudio = useRef<HTMLAudioElement | null>(null);

  /**
   * INITIALIZATION: Load models and setup audio
   */
  useEffect(() => {
    // Initialize alarm audio with a public beep sound
    alarmAudio.current = new Audio('https://assets.mixkit.co/active_storage/sfx/994/994-preview.mp3');
    alarmAudio.current.loop = true;

    const loadModels = async () => {
      try {
        setStatus('loading');
        // Load the required models
        // SsdMobilenetv1 is for high accuracy (used for registration)
        // TinyFaceDetector is for speed (used for real-time tracking)
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);
        setModelsLoaded(true);
        setStatus('idle');
        startVideo(); // Start webcam once models are ready
      } catch (err) {
        console.error('Error loading models:', err);
        alert('Failed to load face-api models. Please check your internet connection.');
      }
    };

    loadModels();

    // Cleanup: Stop alarm if component unmounts
    return () => {
      if (alarmAudio.current) {
        alarmAudio.current.pause();
      }
    };
  }, []);

  /**
   * WEBCAM: Request access to the user's camera
   */
  const startVideo = () => {
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error('Error accessing webcam:', err);
        alert('Could not access camera. Please ensure you have given permission.');
      });
  };

  /**
   * DETECTION LOOP: Triggered when video starts playing
   */
  const handleVideoPlay = () => {
    if (!isDetecting) {
      setIsDetecting(true);
      detectFaces();
    }
  };

  const detectFaces = async () => {
    if (!videoRef.current || !canvasRef.current || !modelsLoaded) return;

    // Match canvas size to video dimensions
    const displaySize = {
      width: videoRef.current.videoWidth,
      height: videoRef.current.videoHeight,
    };
    faceapi.matchDimensions(canvasRef.current, displaySize);

    // Main loop running every DETECTION_INTERVAL
    const intervalId = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;

      // 1. Detect all faces in the current video frame using SsdMobilenetv1 (High Accuracy)
      // This is slower than TinyFaceDetector but much more accurate for recognition
      const detections = await faceapi
        .detectAllFaces(videoRef.current, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
        .withFaceLandmarks()
        .withFaceDescriptors();

      // 2. Resize results to match the display size
      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      
      // 3. Clear previous drawings
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, displaySize.width, displaySize.height);
      }

      if (resizedDetections.length > 0) {
        let isAnyAuthorized = false;
        let recognizedName = '';

        // 4. Process each detected face
        resizedDetections.forEach((detection) => {
          const box = detection.detection.box;
          let label = 'Unknown';
          let color = '#ef4444'; // Red for unknown
          let confidence = (detection.detection.score * 100).toFixed(0);

          // 5. Compare with registered faces
          if (registeredFacesRef.current.length > 0) {
            const faceMatcher = new faceapi.FaceMatcher(
              registeredFacesRef.current.map(f => new faceapi.LabeledFaceDescriptors(f.name, [f.descriptor])),
              DISTANCE_THRESHOLD
            );
            const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
            
            if (bestMatch.label !== 'unknown') {
              label = `${bestMatch.label} (${(100 - bestMatch.distance * 100).toFixed(0)}%)`;
              color = '#22c55e'; // Green for authorized
              isAnyAuthorized = true;
              recognizedName = bestMatch.label;
            } else {
              label = `Unknown (${confidence}%)`;
            }
          } else {
            label = `Face Detected (${confidence}%)`;
          }

          // 6. Draw bounding box and label on canvas
          if (ctx) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.strokeRect(box.x, box.y, box.width, box.height);
            
            // Draw background for label
            ctx.fillStyle = color;
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(box.x, box.y - 30, textWidth + 20, 30);
            
            // Draw label text
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 16px Inter, sans-serif';
            ctx.fillText(label, box.x + 10, box.y - 10);
          }
        });

        // 7. Update system status and alarm
        if (isAnyAuthorized) {
          setStatus('authorized');
          setDetectedName(recognizedName);
          stopAlarm();
        } else {
          setStatus('unknown');
          setDetectedName('Unknown Person');
          if (isAlarmEnabled) playAlarm();
        }
      } else {
        // No faces detected
        setStatus('idle');
        setDetectedName('');
        stopAlarm();
      }
    }, DETECTION_INTERVAL);

    return () => clearInterval(intervalId);
  };

  /**
   * ALARM SYSTEM: Play/Stop the audio alert
   */
  const playAlarm = () => {
    if (alarmAudio.current && alarmAudio.current.paused) {
      alarmAudio.current.play().catch(e => console.warn("Audio play blocked by browser", e));
    }
  };

  const stopAlarm = () => {
    if (alarmAudio.current && !alarmAudio.current.paused) {
      alarmAudio.current.pause();
      alarmAudio.current.currentTime = 0;
    }
  };

  /**
   * REGISTRATION: Capture current face and save descriptor
   */
  const registerFace = async () => {
    if (!videoRef.current || !newName.trim()) return;

    try {
      // Detect single face for registration using the HIGH ACCURACY model (SsdMobilenetv1)
      // This ensures the "face signature" we save is as precise as possible
      const detection = await faceapi
        .detectSingleFace(videoRef.current, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        // Capture a frame for the UI thumbnail
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        canvas.getContext('2d')?.drawImage(videoRef.current, 0, 0);
        const imageData = canvas.toDataURL('image/png');

        const newFace: RegisteredFace = {
          name: newName,
          descriptor: detection.descriptor,
          image: imageData
        };

        setRegisteredFaces(prev => [...prev, newFace]);
        setNewName('');
        setIsRegistering(false);
        alert(`Successfully registered ${newName}!`);
      } else {
        alert('No face detected. Please look directly at the camera and ensure good lighting.');
      }
    } catch (err) {
      console.error('Error registering face:', err);
      alert('Registration failed. Make sure the models are fully loaded.');
    }
  };

  const removeFace = (name: string) => {
    setRegisteredFaces(prev => prev.filter(f => f.name !== name));
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white font-sans selection:bg-orange-500/30">
      {/* Header Section */}
      <header className="p-6 border-b border-white/10 flex justify-between items-center bg-black/50 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-orange-500 rounded-lg flex items-center justify-center shadow-lg shadow-orange-500/20">
            <ShieldCheck className="text-black" size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">FACE GUARD</h1>
            <p className="text-[10px] text-white/40 uppercase tracking-[0.2em]">Security Protocol v2.4</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setIsAlarmEnabled(!isAlarmEnabled)}
            className={`p-2 rounded-full transition-colors ${isAlarmEnabled ? 'text-orange-500 bg-orange-500/10' : 'text-white/40 bg-white/5'}`}
            title={isAlarmEnabled ? "Mute Alarm" : "Unmute Alarm"}
          >
            {isAlarmEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
          </button>
          <button 
            onClick={() => setIsRegistering(true)}
            className="flex items-center gap-2 bg-white text-black px-4 py-2 rounded-full font-semibold text-sm hover:bg-orange-500 hover:text-white transition-all active:scale-95"
          >
            <UserPlus size={16} />
            REGISTER FACE
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Viewport: Video and Canvas */}
        <div className="lg:col-span-2 space-y-6">
          <div className="relative aspect-video bg-white/5 rounded-3xl overflow-hidden border border-white/10 shadow-2xl group">
            <video
              ref={videoRef}
              autoPlay
              muted
              onPlay={handleVideoPlay}
              className="w-full h-full object-cover"
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
            />
            
            {/* Status Overlay */}
            <div className="absolute top-6 left-6 flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full animate-pulse ${
                status === 'authorized' ? 'bg-green-500 shadow-[0_0_12px_rgba(34,197,94,0.6)]' : 
                status === 'unknown' ? 'bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.6)]' : 
                'bg-white/20'
              }`} />
              <span className="text-xs font-mono tracking-widest uppercase opacity-60">
                System: {status === 'loading' ? 'Initializing...' : 'Active'}
              </span>
            </div>

            {/* Scanning Effect Animation */}
            {status === 'idle' && (
              <div className="absolute inset-0 pointer-events-none">
                <div className="w-full h-[2px] bg-orange-500/30 absolute top-0 animate-[scan_3s_ease-in-out_infinite]" />
              </div>
            )}
          </div>

          {/* Real-time Status Bar */}
          <AnimatePresence mode="wait">
            {status !== 'idle' && status !== 'loading' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className={`p-6 rounded-2xl border flex items-center justify-between ${
                  status === 'authorized' 
                    ? 'bg-green-500/10 border-green-500/20 text-green-400' 
                    : 'bg-red-500/10 border-red-500/20 text-red-400'
                }`}
              >
                <div className="flex items-center gap-4">
                  {status === 'authorized' ? <ShieldCheck size={32} /> : <ShieldAlert size={32} />}
                  <div>
                    <h2 className="text-lg font-bold uppercase tracking-tight">
                      {status === 'authorized' ? 'Authorized Person' : 'Unknown Person Detected'}
                    </h2>
                    <p className="text-sm opacity-70">
                      {status === 'authorized' ? `Identity Confirmed: ${detectedName}` : 'Security Alert: Unknown entity detected in secure zone'}
                    </p>
                  </div>
                </div>
                {status === 'unknown' && (
                  <div className="flex items-center gap-2 px-3 py-1 bg-red-500 text-white text-[10px] font-bold rounded-full animate-pulse">
                    ALARM ACTIVE
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Instructions and Accuracy Tips */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white/5 border border-white/10 p-6 rounded-3xl flex items-start gap-4">
              <Info className="text-orange-500 shrink-0" size={24} />
              <div className="text-sm text-white/60 space-y-2">
                <p className="font-bold text-white/80">How it works:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>System uses high-accuracy SSD MobileNet V1 models.</li>
                  <li>Real-time verification runs every 600ms.</li>
                  <li>Recognition is based on 128-point facial geometry.</li>
                </ul>
              </div>
            </div>

            <div className="bg-white/5 border border-white/10 p-6 rounded-3xl flex items-start gap-4">
              <ShieldCheck className="text-green-500 shrink-0" size={24} />
              <div className="text-sm text-white/60 space-y-2">
                <p className="font-bold text-white/80">Tips for Accuracy:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>Ensure your face is well-lit from the front.</li>
                  <li>Look directly at the camera during registration.</li>
                  <li>Remove glasses or hats if they cause issues.</li>
                  <li>Register multiple times in different lighting if needed.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar: Authorized Personnel List */}
        <div className="space-y-6">
          <div className="bg-white/5 rounded-3xl p-6 border border-white/10 h-full">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-sm font-bold uppercase tracking-widest opacity-60">Authorized Personnel</h3>
              <span className="text-[10px] bg-white/10 px-2 py-0.5 rounded-full">{registeredFaces.length}</span>
            </div>

            {registeredFaces.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center space-y-4 opacity-30">
                <Camera size={48} strokeWidth={1} />
                <p className="text-sm">No authorized faces registered.<br/>Add yourself to begin.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {registeredFaces.map((face) => (
                  <div key={face.name} className="group flex items-center gap-4 p-3 bg-white/5 rounded-2xl border border-transparent hover:border-white/10 transition-all">
                    <img src={face.image} alt={face.name} className="w-12 h-12 rounded-xl object-cover border border-white/10" />
                    <div className="flex-1">
                      <p className="text-sm font-semibold">{face.name}</p>
                      <p className="text-[10px] opacity-40 uppercase tracking-tighter">Verified Identity</p>
                    </div>
                    <button 
                      onClick={() => removeFace(face.name)}
                      className="p-2 text-white/20 hover:text-red-500 transition-colors"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Registration Modal Overlay */}
      <AnimatePresence>
        {isRegistering && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsRegistering(false)}
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative bg-[#111] border border-white/10 p-8 rounded-[2rem] w-full max-w-md shadow-2xl"
            >
              <h2 className="text-2xl font-bold mb-2">Register New Face</h2>
              <p className="text-sm text-white/50 mb-6">Position yourself clearly in the camera view and enter your name.</p>
              
              <div className="space-y-4">
                <div>
                  <label className="text-[10px] uppercase tracking-widest opacity-40 mb-2 block">Personnel Name</label>
                  <input 
                    type="text" 
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="Enter name..."
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-orange-500 transition-colors"
                  />
                </div>
                <button 
                  onClick={registerFace}
                  disabled={!newName.trim()}
                  className="w-full bg-orange-500 text-black font-bold py-4 rounded-xl hover:bg-orange-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  <Camera size={20} />
                  CAPTURE & VERIFY
                </button>
                <button 
                  onClick={() => setIsRegistering(false)}
                  className="w-full text-white/40 text-sm font-medium py-2 hover:text-white transition-colors"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes scan {
          0%, 100% { top: 0%; }
          50% { top: 100%; }
        }
      `}} />
    </div>
  );
}
