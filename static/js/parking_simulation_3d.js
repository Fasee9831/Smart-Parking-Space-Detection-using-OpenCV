/**
 * ==============================================================================
 *   SMART VACANTRIX: 3D DIGITAL TWIN PARKING SIMULATION ENGINE (WEBGL / THREE.JS)
 * ==============================================================================
 * 
 * File: static/js/parking_simulation_3d.js
 * Description:
 *   High-fidelity interactive 3D simulation map representing physical parking
 *   stalls, real-time vehicle arrivals/departures, and occupancy status.
 *   Synchronized with live video analytics from live.mp4.
 *
 * Core Architectural Components:
 *   1. Scene & Renderer Initialization: PBR materials, tone mapping, shadows.
 *   2. Procedural Asphalt Ground: Stalls, driving lanes, road markings.
 *   3. 3D Vehicle Generator: Generates low-poly cars with wheels, windows & lights.
 *   4. Real-Time Telemetry Sync: Live polling to reflect video frame detections.
 *   5. Interactive Raycasting: Click & hover detection for parking bays.
 *   6. Camera Viewport Presets: Isometric 3D, Bird's-Eye (Top-Down), Driver Gate View.
 * ==============================================================================
 */

(function () {
  'use strict';

  // Simulation State Container
  const Sim3D = {
    container: null,
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    raycaster: null,
    mouse: null,
    
    // Meshes and objects dictionary
    stallMeshes: {},
    vehicleMeshes: {},
    recommendedArrow: null,
    
    // Lighting references
    ambientLight: null,
    sunLight: null,
    nightLights: [],
    isNightMode: false,
    
    // Polling and sync
    syncTimer: null,
    isInitialized: false,
    animationId: null,
    hoveredStall: null
  };

  /**
   * Initializes the 3D Simulation Canvas inside the Analytics container.
   * @param {string} containerId - DOM ID where the Three.js canvas will mount.
   */
  window.initParkingSimulation3D = function (containerId = 'simulation-3d-canvas-wrap') {
    if (Sim3D.isInitialized) {
      onWindowResize();
      return;
    }

    Sim3D.container = document.getElementById(containerId);
    if (!Sim3D.container) {
      console.warn('[Sim3D] Container element #' + containerId + ' not found.');
      return;
    }

    const width = Sim3D.container.clientWidth || 800;
    const height = Sim3D.container.clientHeight || 460;

    // 1. Initialize Scene
    Sim3D.scene = new THREE.Scene();
    Sim3D.scene.background = new THREE.Color(0x0a0f1d);
    Sim3D.scene.fog = new THREE.FogExp2(0x0a0f1d, 0.009);

    // 2. Initialize Camera (Perspective, wide field of view)
    Sim3D.camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 500);
    setCameraPreset('isometric');

    // 3. Initialize WebGL Renderer
    Sim3D.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    Sim3D.renderer.setSize(width, height);
    Sim3D.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    Sim3D.renderer.shadowMap.enabled = true;
    Sim3D.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    Sim3D.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    Sim3D.renderer.toneMappingExposure = 1.1;
    Sim3D.container.innerHTML = '';
    Sim3D.container.appendChild(Sim3D.renderer.domElement);

    // 4. Initialize Orbit Controls for smooth 3D camera navigation
    if (typeof THREE.OrbitControls !== 'undefined') {
      Sim3D.controls = new THREE.OrbitControls(Sim3D.camera, Sim3D.renderer.domElement);
      Sim3D.controls.enableDamping = true;
      Sim3D.controls.dampingFactor = 0.05;
      Sim3D.controls.maxPolarAngle = Math.PI / 2 - 0.05; // Prevent camera from going beneath floor
      Sim3D.controls.minDistance = 15;
      Sim3D.controls.maxDistance = 120;
    }

    // 5. Initialize Lighting Environment
    setupLighting();

    // 6. Build the Physical Parking Lot Environment (Asphalt, markings, light poles)
    buildParkingLotEnvironment();

    // 7. Raycasting for Mouse Interaction
    Sim3D.raycaster = new THREE.Raycaster();
    Sim3D.mouse = new THREE.Vector2();

    Sim3D.renderer.domElement.addEventListener('mousemove', onMouseMove, false);
    Sim3D.renderer.domElement.addEventListener('click', onMouseClick, false);
    window.addEventListener('resize', onWindowResize, false);

    // 8. Fetch Initial 3D Simulation Data from API
    fetchSimulationData();

    // 9. Start Real-time Telemetry Polling (Syncs with video detection at 1.2s intervals)
    startLiveTelemetrySync();

    // 10. Start Animation Loop
    Sim3D.isInitialized = true;
    animate();
  };

  /**
   * Configures environmental PBR lighting (Sunlight, Ambient, Streetlamps).
   */
  function setupLighting() {
    // Ambient light: Soft cool skylight
    Sim3D.ambientLight = new THREE.AmbientLight(0xdbeafe, 0.65);
    Sim3D.scene.add(Sim3D.ambientLight);

    // Directional Sunlight: Warm overhead illumination casting soft shadows
    Sim3D.sunLight = new THREE.DirectionalLight(0xfffbeb, 1.4);
    Sim3D.sunLight.position.set(35, 60, 40);
    Sim3D.sunLight.castShadow = true;
    Sim3D.sunLight.shadow.mapSize.width = 2048;
    Sim3D.sunLight.shadow.mapSize.height = 2048;
    Sim3D.sunLight.shadow.camera.near = 10;
    Sim3D.sunLight.shadow.camera.far = 160;
    const d = 50;
    Sim3D.sunLight.shadow.camera.left = -d;
    Sim3D.sunLight.shadow.camera.right = d;
    Sim3D.sunLight.shadow.camera.top = d;
    Sim3D.sunLight.shadow.camera.bottom = -d;
    Sim3D.sunLight.shadow.bias = -0.0005;
    Sim3D.scene.add(Sim3D.sunLight);

    // Secondary fill light for balanced contrast
    const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.35);
    fillLight.position.set(-30, 20, -30);
    Sim3D.scene.add(fillLight);
  }

  /**
   * Constructs ground asphalt, stall boundary lines, and driving lane markings.
   */
  function buildParkingLotEnvironment() {
    // 1. Asphalt Ground Plane
    const groundGeo = new THREE.PlaneGeometry(100, 85);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x111827,
      roughness: 0.88,
      metalness: 0.1
    });
    const groundMesh = new THREE.Mesh(groundGeo, groundMat);
    groundMesh.rotation.x = -Math.PI / 2;
    groundMesh.receiveShadow = true;
    Sim3D.scene.add(groundMesh);

    // 2. Central Driving Lane Striping (White dotted centerline)
    for (let x = -40; x <= 40; x += 6) {
      const stripeGeo = new THREE.PlaneGeometry(3.5, 0.4);
      const stripeMat = new THREE.MeshBasicMaterial({ color: 0xf1f5f9 });
      const stripeMesh = new THREE.Mesh(stripeGeo, stripeMat);
      stripeMesh.rotation.x = -Math.PI / 2;
      stripeMesh.position.set(x, 0.02, 0); // Center dividing road
      Sim3D.scene.add(stripeMesh);
    }

    // 3. Directional Road Arrows
    createDrivingArrow(-25, 0.02, 8, 0);
    createDrivingArrow(25, 0.02, -8, Math.PI);

    // 4. Perimeter Curb & Grass Verge
    const curbGeo = new THREE.BoxGeometry(102, 0.6, 2);
    const curbMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.7 });
    
    const northCurb = new THREE.Mesh(curbGeo, curbMat);
    northCurb.position.set(0, 0.3, 41.5);
    Sim3D.scene.add(northCurb);

    const southCurb = new THREE.Mesh(curbGeo, curbMat);
    southCurb.position.set(0, 0.3, -41.5);
    Sim3D.scene.add(southCurb);

    // 5. Overhead Optimal Bay Hologram Arrow
    const arrowGeo = new THREE.ConeGeometry(1.2, 2.4, 4);
    const arrowMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b, wireframe: false });
    Sim3D.recommendedArrow = new THREE.Mesh(arrowGeo, arrowMat);
    Sim3D.recommendedArrow.rotation.x = Math.PI; // Point downwards towards optimal bay
    Sim3D.recommendedArrow.position.set(0, 6, 0);
    Sim3D.recommendedArrow.visible = false;
    Sim3D.scene.add(Sim3D.recommendedArrow);
  }

  /**
   * Helper to generate a directional road arrow.
   */
  function createDrivingArrow(x, y, z, rotation) {
    const arrowGroup = new THREE.Group();
    const shaftGeo = new THREE.PlaneGeometry(1.2, 4);
    const headGeo = new THREE.ConeGeometry(1.8, 2.5, 3);
    const mat = new THREE.MeshBasicMaterial({ color: 0xe2e8f0 });

    const shaft = new THREE.Mesh(shaftGeo, mat);
    shaft.rotation.x = -Math.PI / 2;
    arrowGroup.add(shaft);

    const head = new THREE.Mesh(headGeo, mat);
    head.rotation.x = -Math.PI / 2;
    head.position.z = 2.8;
    arrowGroup.add(head);

    arrowGroup.position.set(x, y, z);
    arrowGroup.rotation.y = rotation;
    Sim3D.scene.add(arrowGroup);
  }

  /**
   * Fetches the 3D manifest and spatial coordinates from the backend.
   */
  async function fetchSimulationData() {
    try {
      const res = await fetch('/api/simulation_3d');
      const data = await res.json();
      if (data.status === 'success' && data.stalls) {
        renderStalls(data.stalls, data.environment.recommended_bay_id);
      }
    } catch (err) {
      console.warn('[Sim3D] Deferred loading simulation state:', err);
    }
  }

  /**
   * Builds 3D stall boundary geometry and generates vehicles for occupied spots.
   */
  function renderStalls(stalls, recommendedId) {
    stalls.forEach(stall => {
      const sid = stall.id;
      const pos = stall.world_position;
      const dim = stall.dimensions;

      // 1. Check if stall group already exists
      let stallGroup = Sim3D.stallMeshes[sid];
      if (!stallGroup) {
        stallGroup = new THREE.Group();
        stallGroup.position.set(pos.x, pos.y, pos.z);
        stallGroup.userData = { stallId: sid, stallData: stall };

        // Floor Pad: Translucent glowing pad
        const padGeo = new THREE.PlaneGeometry(dim.width, dim.length);
        const padMat = new THREE.MeshStandardMaterial({
          color: stall.is_occupied ? 0xef4444 : 0x10b981,
          transparent: true,
          opacity: stall.is_occupied ? 0.35 : 0.55,
          roughness: 0.6
        });
        const padMesh = new THREE.Mesh(padGeo, padMat);
        padMesh.rotation.x = -Math.PI / 2;
        padMesh.receiveShadow = true;
        padMesh.name = 'floorPad';
        stallGroup.add(padMesh);

        // White Stall Boundary Lines (Painted Parking Markings)
        const lineGeo = new THREE.EdgesGeometry(padGeo);
        const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
        const lineMesh = new THREE.LineSegments(lineGeo, lineMat);
        lineMesh.rotation.x = -Math.PI / 2;
        lineMesh.position.y = 0.01;
        stallGroup.add(lineMesh);

        Sim3D.scene.add(stallGroup);
        Sim3D.stallMeshes[sid] = stallGroup;
      }

      // Update Floor Pad Appearance
      const pad = stallGroup.getObjectByName('floorPad');
      if (pad) {
        const isRec = (sid === recommendedId);
        if (isRec) {
          pad.material.color.setHex(0xf59e0b); // Golden amber for optimal space
          pad.material.opacity = 0.85;
        } else if (stall.is_occupied) {
          pad.material.color.setHex(0xe11d48); // Red for occupied
          pad.material.opacity = 0.35;
        } else {
          pad.material.color.setHex(0x10b981); // Emerald green for vacant
          pad.material.opacity = 0.65;
        }
      }

      // Update Vehicle Existence
      if (stall.is_occupied) {
        if (!Sim3D.vehicleMeshes[sid]) {
          const car = createProceduralVehicle(stall.vehicle_profile);
          car.position.set(pos.x, 0.05, pos.z);
          Sim3D.scene.add(car);
          Sim3D.vehicleMeshes[sid] = car;
        }
      } else {
        // Remove parked car if stall is now vacant
        if (Sim3D.vehicleMeshes[sid]) {
          Sim3D.scene.remove(Sim3D.vehicleMeshes[sid]);
          delete Sim3D.vehicleMeshes[sid];
        }
      }

      // Position optimal bay arrow
      if (sid === recommendedId && Sim3D.recommendedArrow) {
        Sim3D.recommendedArrow.position.set(pos.x, 5.5, pos.z);
        Sim3D.recommendedArrow.visible = true;
      }
    });
  }

  /**
   * Procedurally generates a low-poly 3D vehicle model with realistic proportions.
   */
  function createProceduralVehicle(profile) {
    const car = new THREE.Group();
    const colHex = profile ? parseInt(profile.color_hex.replace('#', '0x'), 16) : 0x2563eb;

    // 1. Lower Body Chassis
    const bodyW = profile ? profile.width : 2.4;
    const bodyL = profile ? profile.length : 4.6;
    const bodyH = 0.75;
    
    const bodyGeo = new THREE.BoxGeometry(bodyW, bodyH, bodyL);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: colHex,
      metalness: 0.75,
      roughness: 0.25
    });
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    bodyMesh.position.y = bodyH / 2 + 0.35;
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;
    car.add(bodyMesh);

    // 2. Cabin Roof & Tinted Windows
    const cabinW = bodyW * 0.85;
    const cabinL = bodyL * 0.55;
    const cabinH = 0.65;
    
    const cabinGeo = new THREE.BoxGeometry(cabinW, cabinH, cabinL);
    const cabinMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.1,
      metalness: 0.9,
      transparent: true,
      opacity: 0.92
    });
    const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
    cabinMesh.position.y = bodyH + cabinH / 2 + 0.35;
    cabinMesh.position.z = -0.15;
    cabinMesh.castShadow = true;
    car.add(cabinMesh);

    // 3. Rubber Wheels with Metallic Rims
    const wheelRadius = 0.35;
    const wheelWidth = 0.28;
    const wheelGeo = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 16);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.9 });

    const wheelPositions = [
      [-bodyW / 2, wheelRadius,  bodyL * 0.3],
      [ bodyW / 2, wheelRadius,  bodyL * 0.3],
      [-bodyW / 2, wheelRadius, -bodyL * 0.3],
      [ bodyW / 2, wheelRadius, -bodyL * 0.3]
    ];

    wheelPositions.forEach(pos => {
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(pos[0], pos[1], pos[2]);
      wheel.castShadow = true;
      car.add(wheel);
    });

    // 4. Front Headlights (Warm White Glow)
    const lightGeo = new THREE.BoxGeometry(0.4, 0.15, 0.1);
    const frontLightMat = new THREE.MeshBasicMaterial({ color: 0xfffbeb });
    
    const leftHead = new THREE.Mesh(lightGeo, frontLightMat);
    leftHead.position.set(-bodyW * 0.35, bodyH / 2 + 0.35, bodyL / 2 + 0.02);
    car.add(leftHead);

    const rightHead = new THREE.Mesh(lightGeo, frontLightMat);
    rightHead.position.set(bodyW * 0.35, bodyH / 2 + 0.35, bodyL / 2 + 0.02);
    car.add(rightHead);

    // 5. Rear Taillights (Emissive Crimson)
    const tailMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const leftTail = new THREE.Mesh(lightGeo, tailMat);
    leftTail.position.set(-bodyW * 0.35, bodyH / 2 + 0.35, -bodyL / 2 - 0.02);
    car.add(leftTail);

    const rightTail = new THREE.Mesh(lightGeo, tailMat);
    rightTail.position.set(bodyW * 0.35, bodyH / 2 + 0.35, -bodyL / 2 - 0.02);
    car.add(rightTail);

    return car;
  }

  /**
   * Synchronizes 3D space with live video telemetry stream.
   */
  function startLiveTelemetrySync() {
    if (Sim3D.syncTimer) clearInterval(Sim3D.syncTimer);
    
    Sim3D.syncTimer = setInterval(async () => {
      try {
        const res = await fetch('/api/video_status');
        const data = await res.json();
        if (data.status === 'success' && data.stats) {
          // Re-fetch simulation state when vehicle count alters
          fetchSimulationData();
        }
      } catch (e) {
        // Quiet failover if stream is buffering
      }
    }, 1300);
  }

  /**
   * Sets Camera Viewport Presets: Isometric, Top-Down Bird's-Eye, Driver Entry.
   */
  function setCameraPreset(preset) {
    if (!Sim3D.camera) return;

    if (preset === 'isometric') {
      Sim3D.camera.position.set(38, 42, 52);
      if (Sim3D.controls) Sim3D.controls.target.set(0, 0, 0);
    } else if (preset === 'topdown') {
      Sim3D.camera.position.set(0, 75, 0.1);
      if (Sim3D.controls) Sim3D.controls.target.set(0, 0, 0);
    } else if (preset === 'driver') {
      Sim3D.camera.position.set(-35, 4.5, 0);
      if (Sim3D.controls) Sim3D.controls.target.set(10, 1.5, 0);
    }
    if (Sim3D.controls) Sim3D.controls.update();
  }
  window.setSimCameraPreset = setCameraPreset;

  /**
   * Toggles Day / Night lighting atmosphere.
   */
  window.toggleSimNightMode = function () {
    Sim3D.isNightMode = !Sim3D.isNightMode;
    const btn = document.getElementById('btn-3d-night');

    if (Sim3D.isNightMode) {
      Sim3D.scene.background.setHex(0x02040a);
      Sim3D.ambientLight.intensity = 0.18;
      Sim3D.sunLight.intensity = 0.25;
      Sim3D.sunLight.color.setHex(0x38bdf8);
      if (btn) btn.textContent = '☀️ Day Lighting';
    } else {
      Sim3D.scene.background.setHex(0x0a0f1d);
      Sim3D.ambientLight.intensity = 0.65;
      Sim3D.sunLight.intensity = 1.4;
      Sim3D.sunLight.color.setHex(0xfffbeb);
      if (btn) btn.textContent = '🌙 Night Mode';
    }
  };

  /**
   * Raycasting mouse event to highlight hovered stalls in 3D.
   */
  function onMouseMove(event) {
    const rect = Sim3D.renderer.domElement.getBoundingClientRect();
    Sim3D.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    Sim3D.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    Sim3D.raycaster.setFromCamera(Sim3D.mouse, Sim3D.camera);
    const intersects = Sim3D.raycaster.intersectObjects(Object.values(Sim3D.stallMeshes), true);

    const tooltip = document.getElementById('sim-3d-hover-card');

    if (intersects.length > 0) {
      let group = intersects[0].object;
      while (group.parent && !group.userData.stallData) {
        group = group.parent;
      }

      if (group && group.userData && group.userData.stallData) {
        const stall = group.userData.stallData;
        Sim3D.hoveredStall = stall;

        if (tooltip) {
          tooltip.style.display = 'block';
          tooltip.style.left = `${event.clientX - rect.left + 15}px`;
          tooltip.style.top = `${event.clientY - rect.top - 20}px`;
          const statusCol = stall.is_occupied ? '#e11d48' : '#10b981';
          tooltip.innerHTML = `<b>Bay #${stall.id}</b> • <span style="color:${statusCol}; font-weight:700;">${stall.is_occupied ? 'Occupied' : 'Vacant'}</span><br><span style="color:#94a3b8; font-size:0.7rem;">${stall.zone}</span>`;
        }
        return;
      }
    }

    if (tooltip) tooltip.style.display = 'none';
    Sim3D.hoveredStall = null;
  }

  function onMouseClick() {
    if (Sim3D.hoveredStall && typeof window.onSlotClick === 'function') {
      window.onSlotClick(Sim3D.hoveredStall);
    }
  }

  function onWindowResize() {
    if (!Sim3D.container || !Sim3D.renderer || !Sim3D.camera) return;
    const width = Sim3D.container.clientWidth;
    const height = Sim3D.container.clientHeight || 460;
    Sim3D.camera.aspect = width / height;
    Sim3D.camera.updateProjectionMatrix();
    Sim3D.renderer.setSize(width, height);
  }

  /**
   * Main WebGL Animation Loop (60 FPS).
   */
  function animate() {
    Sim3D.animationId = requestAnimationFrame(animate);

    if (Sim3D.controls) {
      Sim3D.controls.update();
    }

    // Animate Recommended Bay Floating Arrow (Gentle bobbing)
    if (Sim3D.recommendedArrow && Sim3D.recommendedArrow.visible) {
      const time = performance.now() * 0.003;
      Sim3D.recommendedArrow.position.y = 5.2 + Math.sin(time) * 0.4;
      Sim3D.recommendedArrow.rotation.y = time * 0.8;
    }

    if (Sim3D.renderer && Sim3D.scene && Sim3D.camera) {
      Sim3D.renderer.render(Sim3D.scene, Sim3D.camera);
    }
  }

})();
