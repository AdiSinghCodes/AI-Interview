import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import './Avatar.css';

const Avatar = ({ isSpeaking = false, emotion = 'neutral', amplitude = 0 }) => {
  const mountRef = useRef(null);
  const avatarGroupRef = useRef(null);
  const jawRef = useRef(null);
  const mouthRef = useRef(null);
  const upperLipRef = useRef(null);
  const lowerLipRef = useRef(null);
  const leftEyeRef = useRef(null);
  const rightEyeRef = useRef(null);
  const rpmModelRef = useRef(null);
  const amplitudeRef = useRef(amplitude);

  amplitudeRef.current = amplitude;

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 350;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 1000);
    camera.position.set(0, 0.12, 3.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    container.appendChild(renderer.domElement);

    // 2. Realistic Office Lighting Setup
    const ambientLight = new THREE.AmbientLight(0xfff5ea, 1.2);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfffaed, 2.2);
    keyLight.position.set(2.5, 3.5, 4);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x90b0e0, 1.0);
    fillLight.position.set(-3, 1.5, 3);
    scene.add(fillLight);

    const rimLight = new THREE.PointLight(0xffe4ce, 1.8, 10);
    rimLight.position.set(0, 2.5, -2);
    scene.add(rimLight);

    // 3. 3D Office Environment Background
    const officeGroup = new THREE.Group();
    scene.add(officeGroup);

    // Office Wall
    const wallGeo = new THREE.PlaneGeometry(12, 8);
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x1e2638, roughness: 0.8 });
    const wall = new THREE.Mesh(wallGeo, wallMat);
    wall.position.set(0, 0, -4);
    officeGroup.add(wall);

    // Wood Trim Accent
    const woodGeo = new THREE.BoxGeometry(12, 0.15, 0.1);
    const woodMat = new THREE.MeshStandardMaterial({ color: 0x8b5a2b, roughness: 0.4 });
    const woodTrim = new THREE.Mesh(woodGeo, woodMat);
    woodTrim.position.set(0, 1.2, -3.95);
    officeGroup.add(woodTrim);

    // Art Frame Background Accent
    const frameGeo = new THREE.BoxGeometry(1.6, 1.1, 0.05);
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.3 });
    const artFrame = new THREE.Mesh(frameGeo, frameMat);
    artFrame.position.set(-2.2, 0.6, -3.9);
    officeGroup.add(artFrame);

    const artCanvasGeo = new THREE.PlaneGeometry(1.4, 0.9);
    const artCanvasMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const artCanvas = new THREE.Mesh(artCanvasGeo, artCanvasMat);
    artCanvas.position.set(-2.2, 0.6, -3.87);
    officeGroup.add(artCanvas);

    // 4. Main Human Avatar Root Group
    const avatarGroup = new THREE.Group();
    avatarGroupRef.current = avatarGroup;
    avatarGroup.position.set(0, -0.65, 0);
    scene.add(avatarGroup);

    // Procedural 3D Human Executive Model (Instant 0ms Load)
    const humanGroup = new THREE.Group();
    avatarGroup.add(humanGroup);

    // Realistic Skin & Lip Materials
    const skinMat = new THREE.MeshStandardMaterial({
      color: 0xe5b89a, // Realistic natural warm skin tone
      roughness: 0.55,
      metalness: 0.05,
    });

    const lipMat = new THREE.MeshStandardMaterial({
      color: 0xc8746c, // Natural lip shade
      roughness: 0.4,
    });

    // Anatomical Human Head Shape
    const headGeo = new THREE.SphereGeometry(0.72, 64, 64);
    headGeo.scale(0.88, 1.12, 0.92);
    const headMesh = new THREE.Mesh(headGeo, skinMat);
    headMesh.position.set(0, 1.1, 0);
    humanGroup.add(headMesh);

    // Defined Human Jawline & Chin
    const chinGeo = new THREE.BoxGeometry(0.48, 0.4, 0.52);
    const chinMesh = new THREE.Mesh(chinGeo, skinMat);
    chinMesh.position.set(0, 0.68, 0.18);
    humanGroup.add(chinMesh);

    // Nose Group
    const noseGroup = new THREE.Group();
    noseGroup.position.set(0, 1.08, 0.65);

    const noseBridgeGeo = new THREE.CylinderGeometry(0.035, 0.065, 0.32, 16);
    const noseBridge = new THREE.Mesh(noseBridgeGeo, skinMat);
    noseBridge.rotation.x = -0.22;
    noseGroup.add(noseBridge);

    const noseTipGeo = new THREE.SphereGeometry(0.075, 16, 16);
    const noseTip = new THREE.Mesh(noseTipGeo, skinMat);
    noseTip.position.set(0, -0.13, 0.06);
    noseGroup.add(noseTip);
    humanGroup.add(noseGroup);

    // Executive Hair Style (Dark Brown Side-Part)
    const hairMat = new THREE.MeshStandardMaterial({ color: 0x221810, roughness: 0.6, metalness: 0.1 });
    const hairTopGeo = new THREE.SphereGeometry(0.72, 32, 32);
    hairTopGeo.scale(0.92, 0.65, 0.96);
    const hairTop = new THREE.Mesh(hairTopGeo, hairMat);
    hairTop.position.set(0, 1.48, -0.04);
    humanGroup.add(hairTop);

    const hairSwoopGeo = new THREE.ConeGeometry(0.45, 0.8, 16);
    const hairSwoop = new THREE.Mesh(hairSwoopGeo, hairMat);
    hairSwoop.rotation.z = -0.7;
    hairSwoop.rotation.x = -0.3;
    hairSwoop.position.set(0.2, 1.52, 0.22);
    humanGroup.add(hairSwoop);

    // Realistic Human Eyes
    const eyeWhiteMat = new THREE.MeshBasicMaterial({ color: 0xfafafa });
    const irisMat = new THREE.MeshStandardMaterial({ color: 0x2b4c7e, roughness: 0.2 });
    const pupilMat = new THREE.MeshBasicMaterial({ color: 0x050505 });

    const createHumanEye = (isLeft) => {
      const eyeRoot = new THREE.Group();
      const xPos = isLeft ? -0.26 : 0.26;
      eyeRoot.position.set(xPos, 1.15, 0.56);

      const globeGeo = new THREE.SphereGeometry(0.1, 32, 32);
      const globe = new THREE.Mesh(globeGeo, eyeWhiteMat);
      eyeRoot.add(globe);

      const irisGeo = new THREE.SphereGeometry(0.055, 16, 16);
      const iris = new THREE.Mesh(irisGeo, irisMat);
      iris.position.set(0, 0, 0.08);
      eyeRoot.add(iris);

      const pupilGeo = new THREE.SphereGeometry(0.028, 16, 16);
      const pupil = new THREE.Mesh(pupilGeo, pupilMat);
      pupil.position.set(0, 0, 0.098);
      eyeRoot.add(pupil);

      const browGeo = new THREE.BoxGeometry(0.22, 0.035, 0.06);
      const browMat = new THREE.MeshStandardMaterial({ color: 0x221810, roughness: 0.8 });
      const brow = new THREE.Mesh(browGeo, browMat);
      brow.position.set(0, 0.14, 0.04);
      brow.rotation.z = isLeft ? 0.08 : -0.08;
      eyeRoot.add(brow);

      return eyeRoot;
    };

    const leftEye = createHumanEye(true);
    const rightEye = createHumanEye(false);
    humanGroup.add(leftEye);
    humanGroup.add(rightEye);
    leftEyeRef.current = leftEye;
    rightEyeRef.current = rightEye;

    // Ears
    const earGeo = new THREE.SphereGeometry(0.12, 16, 16);
    earGeo.scale(0.4, 0.8, 0.6);
    const leftEar = new THREE.Mesh(earGeo, skinMat);
    leftEar.position.set(-0.64, 1.08, 0.05);
    humanGroup.add(leftEar);

    const rightEar = new THREE.Mesh(earGeo, skinMat);
    rightEar.position.set(0.64, 1.08, 0.05);
    humanGroup.add(rightEar);

    // Mouth & Jaw for Lip Sync
    const jawGroup = new THREE.Group();
    jawGroup.position.set(0, 0.82, 0.45);
    humanGroup.add(jawGroup);
    jawRef.current = jawGroup;

    const upperLipGeo = new THREE.BoxGeometry(0.26, 0.04, 0.12);
    const upperLip = new THREE.Mesh(upperLipGeo, lipMat);
    upperLip.position.set(0, 0.02, 0.18);
    jawGroup.add(upperLip);
    upperLipRef.current = upperLip;

    const lowerLipGeo = new THREE.BoxGeometry(0.24, 0.045, 0.12);
    const lowerLip = new THREE.Mesh(lowerLipGeo, lipMat);
    lowerLip.position.set(0, -0.03, 0.18);
    jawGroup.add(lowerLip);
    lowerLipRef.current = lowerLip;

    const cavityGeo = new THREE.BoxGeometry(0.2, 0.08, 0.1);
    const cavityMat = new THREE.MeshBasicMaterial({ color: 0x110505 });
    const cavity = new THREE.Mesh(cavityGeo, cavityMat);
    cavity.position.set(0, -0.01, 0.12);
    jawGroup.add(cavity);

    // Suit, Shirt & Tie
    const suitGroup = new THREE.Group();
    humanGroup.add(suitGroup);

    const neckGeo = new THREE.CylinderGeometry(0.22, 0.25, 0.4, 32);
    const neck = new THREE.Mesh(neckGeo, skinMat);
    neck.position.set(0, 0.42, 0.02);
    suitGroup.add(neck);

    const shirtGeo = new THREE.ConeGeometry(0.65, 0.8, 32);
    const shirtMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.3 });
    const shirt = new THREE.Mesh(shirtGeo, shirtMat);
    shirt.position.set(0, 0.05, 0.02);
    suitGroup.add(shirt);

    const tieGeo = new THREE.BoxGeometry(0.12, 0.55, 0.08);
    const tieMat = new THREE.MeshStandardMaterial({ color: 0x1d4ed8, roughness: 0.4 });
    const tie = new THREE.Mesh(tieGeo, tieMat);
    tie.position.set(0, 0.08, 0.3);
    suitGroup.add(tie);

    const jacketGeo = new THREE.BoxGeometry(1.4, 0.8, 0.7);
    const jacketMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.5 });
    const jacket = new THREE.Mesh(jacketGeo, jacketMat);
    jacket.position.set(0, -0.25, 0.02);
    suitGroup.add(jacket);

    // Try loading RPM GLB model gracefully in background if available
    try {
      const loader = new GLTFLoader();
      const rpmAvatarUrl = 'https://models.readyplayer.me/6460d35c74ae38612089a445.glb';
      loader.load(
        rpmAvatarUrl,
        (gltf) => {
          const rpmModel = gltf.scene;
          rpmModel.scale.set(1.15, 1.15, 1.15);
          rpmModel.position.set(0, -1.45, 0);
          
          humanGroup.visible = false;
          avatarGroup.add(rpmModel);
          rpmModelRef.current = rpmModel;
        },
        undefined,
        () => {
          // Keep procedural 3D human model active
        }
      );
    } catch(e) {
      // Keep procedural model
    }

    // Render Animation Loop
    let clock = new THREE.Clock();
    let animId;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();
      const currentAmp = amplitudeRef.current;

      // Natural Subtle Breathing & Head Sway
      avatarGroup.position.y = -0.65 + Math.sin(elapsedTime * 1.5) * 0.02;
      avatarGroup.rotation.y = Math.sin(elapsedTime * 0.6) * 0.03;
      avatarGroup.rotation.x = Math.sin(elapsedTime * 0.9) * 0.015;

      // Human Audio Lip Sync Animation
      if (jawRef.current) {
        const jawOpenAmount = isSpeaking ? currentAmp * 0.18 : 0;
        jawRef.current.position.y = THREE.MathUtils.lerp(jawRef.current.position.y, 0.82 - jawOpenAmount, 0.35);
      }

      if (lowerLipRef.current && upperLipRef.current) {
        const lipSeparation = isSpeaking ? currentAmp * 0.08 : 0;
        lowerLipRef.current.position.y = THREE.MathUtils.lerp(lowerLipRef.current.position.y, -0.03 - lipSeparation, 0.4);
        upperLipRef.current.position.y = THREE.MathUtils.lerp(upperLipRef.current.position.y, 0.02 + lipSeparation * 0.4, 0.4);
      }

      // Ready Player Me Model Morph Targets
      if (rpmModelRef.current) {
        rpmModelRef.current.traverse((child) => {
          if (child.isMesh && child.morphTargetDictionary && child.morphTargetInfluences) {
            const jawOpenIdx = child.morphTargetDictionary['jawOpen'] ?? child.morphTargetDictionary['viseme_aa'];
            const mouthOpenIdx = child.morphTargetDictionary['mouthOpen'] ?? child.morphTargetDictionary['viseme_O'];
            
            if (jawOpenIdx !== undefined) {
              child.morphTargetInfluences[jawOpenIdx] = THREE.MathUtils.lerp(
                child.morphTargetInfluences[jawOpenIdx],
                isSpeaking ? currentAmp * 0.85 : 0,
                0.35
              );
            }
            if (mouthOpenIdx !== undefined) {
              child.morphTargetInfluences[mouthOpenIdx] = THREE.MathUtils.lerp(
                child.morphTargetInfluences[mouthOpenIdx],
                isSpeaking ? currentAmp * 0.65 : 0,
                0.35
              );
            }
          }
        });
      }

      // Natural Eye Blinking
      const blinkCycle = elapsedTime % 4.5;
      const isBlinking = blinkCycle > 4.35 && blinkCycle < 4.45;
      const eyeScaleY = isBlinking ? 0.08 : 1.0;

      if (leftEyeRef.current) leftEyeRef.current.scale.y = THREE.MathUtils.lerp(leftEyeRef.current.scale.y, eyeScaleY, 0.6);
      if (rightEyeRef.current) rightEyeRef.current.scale.y = THREE.MathUtils.lerp(rightEyeRef.current.scale.y, eyeScaleY, 0.6);

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!container) return;
      const newW = container.clientWidth;
      const newH = container.clientHeight;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animId);
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [isSpeaking]);

  return (
    <div className="avatar-container animate-fade-in">
      <div className="office-background-overlay" />
      <div className="three-canvas-wrapper" ref={mountRef} />
      {isSpeaking && <div className="speaking-indicator-bar" />}
    </div>
  );
};

export default Avatar;