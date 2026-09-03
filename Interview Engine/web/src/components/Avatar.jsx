import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import './Avatar.css'

// Procedural head — no GLB, no network. Amplitude drives the jaw/lips.
export default function Avatar({ speaking = false, amplitude = 0 }) {
  const mountRef = useRef(null)
  const speakingRef = useRef(speaking)
  const ampRef = useRef(amplitude)
  speakingRef.current = speaking
  ampRef.current = amplitude

  useEffect(() => {
    const container = mountRef.current
    if (!container) return
    const w = container.clientWidth || 420
    const h = container.clientHeight || 360

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 100)
    camera.position.set(0, 0.15, 3.7)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(w, h)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.replaceChildren(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xfff3e6, 1.2))
    const key = new THREE.DirectionalLight(0xfffaed, 2.1)
    key.position.set(2.5, 3.5, 4)
    scene.add(key)
    const fill = new THREE.DirectionalLight(0x94b4e6, 0.9)
    fill.position.set(-3, 1.5, 3)
    scene.add(fill)

    const person = new THREE.Group()
    person.position.set(0, -0.55, 0)
    scene.add(person)

    const skin = new THREE.MeshStandardMaterial({ color: 0xe6b89c, roughness: 0.55 })
    const lipMat = new THREE.MeshStandardMaterial({ color: 0xc8746c, roughness: 0.4 })
    const hairMat = new THREE.MeshStandardMaterial({ color: 0x241a12, roughness: 0.6 })

    const headGeo = new THREE.SphereGeometry(0.72, 48, 48)
    headGeo.scale(0.9, 1.12, 0.94)
    const head = new THREE.Mesh(headGeo, skin)
    head.position.y = 1.1
    person.add(head)

    const chin = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.38, 0.5), skin)
    chin.position.set(0, 0.7, 0.16)
    person.add(chin)

    const hair = new THREE.Mesh(new THREE.SphereGeometry(0.73, 32, 32), hairMat)
    hair.geometry.scale(0.94, 0.62, 0.98)
    hair.position.set(0, 1.46, -0.04)
    person.add(hair)

    const nose = new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 16), skin)
    nose.position.set(0, 1.02, 0.66)
    person.add(nose)

    const eyes = []
    for (const x of [-0.26, 0.26]) {
      const g = new THREE.Group()
      g.position.set(x, 1.16, 0.55)
      const white = new THREE.Mesh(new THREE.SphereGeometry(0.1, 20, 20), new THREE.MeshBasicMaterial({ color: 0xfafafa }))
      const iris = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 16), new THREE.MeshStandardMaterial({ color: 0x2b4c7e }))
      iris.position.z = 0.08
      g.add(white, iris)
      person.add(g)
      eyes.push(g)
    }

    const jaw = new THREE.Group()
    jaw.position.set(0, 0.84, 0.46)
    person.add(jaw)
    const upperLip = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.04, 0.12), lipMat)
    upperLip.position.set(0, 0.02, 0.18)
    const lowerLip = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.045, 0.12), lipMat)
    lowerLip.position.set(0, -0.03, 0.18)
    const mouthDark = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.08, 0.1), new THREE.MeshBasicMaterial({ color: 0x120606 }))
    mouthDark.position.set(0, -0.01, 0.12)
    jaw.add(upperLip, lowerLip, mouthDark)

    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.24, 0.4, 24), skin)
    neck.position.set(0, 0.42, 0.02)
    person.add(neck)
    const shirt = new THREE.Mesh(new THREE.ConeGeometry(0.66, 0.85, 28), new THREE.MeshStandardMaterial({ color: 0xf1f5f9, roughness: 0.4 }))
    shirt.position.set(0, 0.02, 0.02)
    person.add(shirt)
    const jacket = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.8, 0.7), new THREE.MeshStandardMaterial({ color: 0x101a2e, roughness: 0.5 }))
    jacket.position.set(0, -0.28, 0.02)
    person.add(jacket)

    const clock = new THREE.Clock()
    let raf
    const animate = () => {
      raf = requestAnimationFrame(animate)
      const t = clock.getElapsedTime()
      const amp = speakingRef.current ? ampRef.current : 0

      person.position.y = -0.55 + Math.sin(t * 1.4) * 0.02
      person.rotation.y = Math.sin(t * 0.5) * 0.04

      const open = amp * 0.16
      jaw.position.y = THREE.MathUtils.lerp(jaw.position.y, 0.84 - open, 0.4)
      lowerLip.position.y = THREE.MathUtils.lerp(lowerLip.position.y, -0.03 - amp * 0.07, 0.4)
      upperLip.position.y = THREE.MathUtils.lerp(upperLip.position.y, 0.02 + amp * 0.03, 0.4)

      const blink = t % 4.4 > 4.3 ? 0.1 : 1
      eyes.forEach((e) => (e.scale.y = THREE.MathUtils.lerp(e.scale.y, blink, 0.5)))

      renderer.render(scene, camera)
    }
    animate()

    const onResize = () => {
      const nw = container.clientWidth
      const nh = container.clientHeight
      camera.aspect = nw / nh
      camera.updateProjectionMatrix()
      renderer.setSize(nw, nh)
    }
    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      cancelAnimationFrame(raf)
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])

  return (
    <div className="avatar-container">
      <div className="office-background-overlay" />
      <div className="three-canvas-wrapper" ref={mountRef} />
      {speaking && <div className="speaking-indicator-bar" />}
    </div>
  )
}
