// 4-Way Traffic Junction Simulation Canvas Engine
class JunctionSimulator {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.vehicles = [];
    this.currentPhase = 'PHASE_A_NS'; // 'PHASE_A_NS' or 'PHASE_B_EW'
    this.isEmergency = false;
    this.emergencyDir = null;
    this.spawnRate = 0.05;
    this.lastSpawn = Date.now();
    this.uniqueCrossed = { north: 18, south: 14, east: 5, west: 7 };
    this.phaseTimeRemaining = 35;
    this.lastPhaseTick = Date.now();
    this.isControlledExternally = false;
    this.minGreenTime = 15;
    this.maxRedWait = 40;
    this.queueThreshold = 7;
    this.yellowDuration = 3;
    this.allocatedGreen = { PHASE_A_NS: 35, PHASE_B_EW: 25 };
    this.currentPhaseElapsed = 0;
    this.redPhaseWaitTime = 0;
    this.transitionState = null;
    this.transitionTimeRemaining = 0;
    this.retiringPhase = null;
    this.nextPhase = null;
    this.isManualOverride = false;

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.init();
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width || 700;
    this.canvas.height = rect.height || 380;
    this.cx = this.canvas.width / 2;
    this.cy = this.canvas.height / 2;
    this.roadWidth = 140;
  }

  init() {
    // Initial vehicles
    for (let i = 0; i < 8; i++) {
      this.spawnRandomVehicle(true);
    }
    this.loop();
  }

  setPhase(phase, isEmergency = false, emergencyDir = null) {
    if (this.currentPhase !== phase || this.isEmergency !== isEmergency) {
      this.currentPhase = phase;
      this.isEmergency = isEmergency;
      this.emergencyDir = emergencyDir;
      this.transitionState = null;
      this.currentPhaseElapsed = 0;
      this.redPhaseWaitTime = 0;
    }
  }

  triggerManualOverride(phase) {
    this.isEmergency = false;
    this.emergencyDir = null;
    this.isManualOverride = true;
    this.currentPhase = phase;
    this.transitionState = null;
    this.currentPhaseElapsed = 0;
    this.redPhaseWaitTime = 0;
    this.phaseTimeRemaining = (phase === 'PEDESTRIAN' ? 15 : (phase === 'PHASE_A_NS' ? 35 : 25));
    if (this.allocatedGreen[phase]) {
      this.phaseTimeRemaining = this.allocatedGreen[phase];
    }
    const inViewQueue = this.getInViewQueue();
    const signals = this.getCurrentSignals();
    this.updateSPaTUI(inViewQueue, signals);
  }

  triggerEmergencyPreempt(dir = 'south') {
    this.isEmergency = true;
    this.isManualOverride = false;
    this.emergencyDir = dir;
    this.transitionState = null;
    this.phaseTimeRemaining = 25;
    this.spawnEmergency(dir);
    const inViewQueue = this.getInViewQueue();
    const signals = this.getCurrentSignals();
    this.updateSPaTUI(inViewQueue, signals);
  }

  triggerPhaseSwitch(nextPhase) {
    if (this.yellowDuration > 0) {
      this.transitionState = 'YELLOW';
      this.transitionTimeRemaining = this.yellowDuration;
      this.retiringPhase = this.currentPhase;
      this.nextPhase = nextPhase;
    } else {
      this.transitionState = null;
      this.currentPhase = nextPhase;
      this.currentPhaseElapsed = 0;
      this.redPhaseWaitTime = 0;
      this.phaseTimeRemaining = this.allocatedGreen[this.currentPhase] || (this.currentPhase === 'PHASE_A_NS' ? 35 : 25);
    }
  }

  getInViewQueue() {
    const inViewQueue = { north: 0, south: 0, east: 0, west: 0 };
    for (const v of this.vehicles) {
      if (!v.crossed && inViewQueue[v.direction] !== undefined) {
        inViewQueue[v.direction]++;
      }
    }
    return inViewQueue;
  }

  getCurrentSignals() {
    const signals = { north: 'RED', south: 'RED', east: 'RED', west: 'RED' };
    if (this.isEmergency) {
      const eDir = this.emergencyDir || 'south';
      signals[eDir] = 'GREEN';
    } else if (this.transitionState === 'YELLOW') {
      const ret = this.retiringPhase || this.currentPhase;
      if (ret === 'PHASE_A_NS') {
        signals.north = 'YELLOW';
        signals.south = 'YELLOW';
      } else if (ret === 'PHASE_B_EW') {
        signals.east = 'YELLOW';
        signals.west = 'YELLOW';
      }
    } else if (this.currentPhase === 'PHASE_A_NS') {
      signals.north = 'GREEN';
      signals.south = 'GREEN';
    } else if (this.currentPhase === 'PHASE_B_EW') {
      signals.east = 'GREEN';
      signals.west = 'GREEN';
    }
    return signals;
  }

  spawnRandomVehicle(initial = false) {
    const dirs = ['north', 'south', 'east', 'west'];
    const dir = dirs[Math.floor(Math.random() * dirs.length)];
    const types = ['car', 'car', 'bus', 'bike', 'truck'];
    const type = types[Math.floor(Math.random() * types.length)];
    this.spawnVehicle(dir, type, initial);
  }

  spawnEmergency(dir = 'south') {
    this.spawnVehicle(dir, 'ambulance', false, true);
  }

  spawnVehicle(direction, type, initial = false, isEmergency = false) {
    const laneOffset = 30; // offset from road center
    let x, y, vx = 0, vy = 0;
    const speed = isEmergency ? 3.5 : (type === 'bike' ? 2.5 : (type === 'bus' ? 1.4 : 2.0));

    switch (direction) {
      case 'north': // moving south
        x = this.cx - laneOffset;
        y = initial ? Math.random() * (this.cy - 100) : -40;
        vy = speed;
        break;
      case 'south': // moving north
        x = this.cx + laneOffset;
        y = initial ? this.cy + 100 + Math.random() * 150 : this.canvas.height + 40;
        vy = -speed;
        break;
      case 'east': // moving west
        x = initial ? this.cx + 100 + Math.random() * 200 : this.canvas.width + 40;
        y = this.cy - laneOffset;
        vx = -speed;
        break;
      case 'west': // moving east
        x = initial ? Math.random() * (this.cx - 100) : -40;
        y = this.cy + laneOffset;
        vx = speed;
        break;
    }

    const colors = {
      car: '#3b82f6',
      bus: '#f59e0b',
      bike: '#10b981',
      truck: '#8b5cf6',
      ambulance: '#ef4444'
    };

    const stopLineNorth = this.cy - this.roadWidth / 2 - 15;
    const stopLineSouth = this.cy + this.roadWidth / 2 + 15;
    const stopLineWest = this.cx - this.roadWidth / 2 - 15;
    const stopLineEast = this.cx + this.roadWidth / 2 + 15;

    let crossed = false;
    if (initial) {
      if (direction === 'north' && y >= stopLineNorth) crossed = true;
      else if (direction === 'south' && y <= stopLineSouth) crossed = true;
      else if (direction === 'west' && x >= stopLineWest) crossed = true;
      else if (direction === 'east' && x <= stopLineEast) crossed = true;
    }

    this.vehicles.push({
      x, y, vx, vy,
      baseSpeed: speed,
      direction,
      type,
      color: colors[type],
      isEmergency,
      width: type === 'bike' ? 10 : (type === 'bus' || type === 'truck' ? 24 : 16),
      length: type === 'bike' ? 18 : (type === 'bus' || type === 'truck' ? 44 : 28),
      crossed
    });
  }

  loop() {
    this.update();
    this.draw();
    requestAnimationFrame(() => this.loop());
  }

  update() {
    // Spawning logic
    if (Math.random() < this.spawnRate && this.vehicles.length < 25) {
      this.spawnRandomVehicle();
    }

    const stopLineNorth = this.cy - this.roadWidth / 2 - 15;
    const stopLineSouth = this.cy + this.roadWidth / 2 + 15;
    const stopLineWest = this.cx - this.roadWidth / 2 - 15;
    const stopLineEast = this.cx + this.roadWidth / 2 + 15;

    const signals = this.getCurrentSignals();

    for (let i = this.vehicles.length - 1; i >= 0; i--) {
      const v = this.vehicles[i];
      let canMove = true;

      // Check traffic light signal
      let isGreen = false;
      const appSig = signals[v.direction];
      if (this.isEmergency && v.isEmergency) {
        isGreen = true;
      } else if (appSig === 'GREEN') {
        isGreen = true;
      }

      // Stop at intersection if not green (red or yellow)
      if (!isGreen) {
        if (v.direction === 'north' && v.y < stopLineNorth && v.y + v.vy >= stopLineNorth) {
          canMove = false;
        } else if (v.direction === 'south' && v.y > stopLineSouth && v.y + v.vy <= stopLineSouth) {
          canMove = false;
        } else if (v.direction === 'west' && v.x < stopLineWest && v.x + v.vx >= stopLineWest) {
          canMove = false;
        } else if (v.direction === 'east' && v.x > stopLineEast && v.x + v.vx <= stopLineEast) {
          canMove = false;
        }
      }

      // Collision avoidance with vehicle ahead
      for (let j = 0; j < this.vehicles.length; j++) {
        if (i === j) continue;
        const o = this.vehicles[j];
        if (o.direction === v.direction) {
          if (v.direction === 'north' && o.y > v.y && o.y - v.y < v.length + 15) canMove = false;
          if (v.direction === 'south' && o.y < v.y && v.y - o.y < v.length + 15) canMove = false;
          if (v.direction === 'west' && o.x > v.x && o.x - v.x < v.length + 15) canMove = false;
          if (v.direction === 'east' && o.x < v.x && v.x - o.x < v.length + 15) canMove = false;
        }
      }

      if (canMove) {
        v.x += v.vx;
        v.y += v.vy;
      }

      // Check if vehicle crossed the stop line
      if (!v.crossed) {
        let hasCrossed = false;
        if (v.direction === 'north' && v.y >= stopLineNorth) hasCrossed = true;
        else if (v.direction === 'south' && v.y <= stopLineSouth) hasCrossed = true;
        else if (v.direction === 'west' && v.x >= stopLineWest) hasCrossed = true;
        else if (v.direction === 'east' && v.x <= stopLineEast) hasCrossed = true;

        if (hasCrossed) {
          v.crossed = true;
          if (this.uniqueCrossed && this.uniqueCrossed[v.direction] !== undefined) {
            this.uniqueCrossed[v.direction]++;
          }
        }
      }

      // Remove vehicles that left screen
      if (v.x < -60 || v.x > this.canvas.width + 60 || v.y < -60 || v.y > this.canvas.height + 60) {
        this.vehicles.splice(i, 1);
      }
    }

    const inViewQueue = this.getInViewQueue();

    // Signal switching cycle and adaptive logic
    if (this.canvas.id === 'junctionCanvas') {
      const now = Date.now();
      if (now - this.lastPhaseTick >= 1000) {
        this.lastPhaseTick = now;
        this.tickPhaseLogic(inViewQueue);
      }
      this.updateSPaTUI(inViewQueue, signals);

      // Sync state to secondary canvas if present
      if (window.junctionSim2 && window.junctionSim2 !== this) {
        window.junctionSim2.currentPhase = this.currentPhase;
        window.junctionSim2.isEmergency = this.isEmergency;
        window.junctionSim2.emergencyDir = this.emergencyDir;
        window.junctionSim2.transitionState = this.transitionState;
        window.junctionSim2.retiringPhase = this.retiringPhase;
        window.junctionSim2.nextPhase = this.nextPhase;
      }
    }
  }

  tickPhaseLogic(inViewQueue) {
    // 1. If currently in transition (e.g. Yellow clearance)
    if (this.transitionState === 'YELLOW') {
      this.transitionTimeRemaining--;
      if (this.transitionTimeRemaining <= 0) {
        this.transitionState = null;
        this.currentPhase = this.nextPhase || (this.currentPhase === 'PHASE_A_NS' ? 'PHASE_B_EW' : 'PHASE_A_NS');
        this.currentPhaseElapsed = 0;
        this.redPhaseWaitTime = 0;
        this.phaseTimeRemaining = this.allocatedGreen[this.currentPhase] || (this.currentPhase === 'PHASE_A_NS' ? 35 : 25);
      }
      return;
    }

    // 2. If in Emergency Preemption
    if (this.isEmergency) {
      this.phaseTimeRemaining--;
      const hasAmbulance = this.vehicles.some(v => v.isEmergency);
      if (this.phaseTimeRemaining <= 0 || !hasAmbulance) {
        this.isEmergency = false;
        this.emergencyDir = null;
        this.currentPhase = 'PHASE_A_NS';
        this.currentPhaseElapsed = 0;
        this.redPhaseWaitTime = 0;
        this.phaseTimeRemaining = this.allocatedGreen['PHASE_A_NS'] || 35;
      }
      return;
    }

    // 3. If in Manual Override (Force N-S, Force E-W, Pedestrian)
    if (this.isManualOverride) {
      this.phaseTimeRemaining--;
      if (this.phaseTimeRemaining <= 0) {
        this.isManualOverride = false;
        const next = this.currentPhase === 'PHASE_A_NS' ? 'PHASE_B_EW' : 'PHASE_A_NS';
        this.triggerPhaseSwitch(next);
      }
      return;
    }

    // 4. Automatic Adaptive Switching Logic
    const isPhaseA = this.currentPhase === 'PHASE_A_NS';
    const greenQueue = isPhaseA
      ? (inViewQueue.north + inViewQueue.south)
      : (inViewQueue.east + inViewQueue.west);
    const redQueue = isPhaseA
      ? (inViewQueue.east + inViewQueue.west)
      : (inViewQueue.north + inViewQueue.south);
    const nextPhase = isPhaseA ? 'PHASE_B_EW' : 'PHASE_A_NS';

    // If all four approaches have the same in-view queue count,
    // use the existing normal signal cycle and timing without adaptive queue-triggered exceptions.
    const allEqual = (
      inViewQueue.north === inViewQueue.south &&
      inViewQueue.south === inViewQueue.east &&
      inViewQueue.east === inViewQueue.west
    );

    if (allEqual) {
      this.phaseTimeRemaining--;
      this.currentPhaseElapsed++;
      if (this.phaseTimeRemaining <= 0) {
        this.triggerPhaseSwitch(nextPhase);
      }
      return;
    }

    // When queues differ, execute adaptive rules
    this.currentPhaseElapsed++;
    this.phaseTimeRemaining--;

    // Accumulate wait time for red approaches
    if (redQueue > 0) {
      this.redPhaseWaitTime++;
    } else {
      this.redPhaseWaitTime = 0;
    }

    const minGreen = this.minGreenTime || 15;
    const hasMetMinGreen = this.currentPhaseElapsed >= minGreen;

    // Rule: If there is no queued vehicle waiting for red phase,
    // do not switch away from a phase that still has traffic to serve.
    if (redQueue === 0 && greenQueue > 0) {
      if (this.phaseTimeRemaining <= 0) {
        this.phaseTimeRemaining = 1;
      }
      return;
    }

    // Rule: If current phase has no queued traffic and other phase does,
    // switch to serve waiting traffic, subject to existing safe transition rules
    if (greenQueue === 0 && redQueue > 0) {
      if (hasMetMinGreen) {
        this.triggerPhaseSwitch(nextPhase);
        return;
      }
    }

    // Rule: When red approaches reach the 7-vehicle queue threshold,
    // provided current phase has met its existing minimum green time.
    if (redQueue >= this.queueThreshold && hasMetMinGreen) {
      this.triggerPhaseSwitch(nextPhase);
      return;
    }

    // Rule: If red approaches do not reach threshold, switch when maximum wait time reaches 40s
    if (this.redPhaseWaitTime >= this.maxRedWait && hasMetMinGreen) {
      this.triggerPhaseSwitch(nextPhase);
      return;
    }

    // Normal countdown expiration
    if (this.phaseTimeRemaining <= 0) {
      this.triggerPhaseSwitch(nextPhase);
      return;
    }
  }

  updateSPaTUI(inViewQueue, signals) {
    if (this.canvas.id !== 'junctionCanvas') return;

    // Approach rows in Signal Phase & Timing table
    ['north', 'south', 'east', 'west'].forEach(dir => {
      const row = document.getElementById(`row-app-${dir}`);
      if (row) {
        const uniqueCell = row.querySelector('.cell-unique');
        const queueCell = row.querySelector('.cell-queue');
        const signalCell = row.querySelector('.cell-signal');
        const timeCell = row.querySelector('.cell-time');

        if (uniqueCell) uniqueCell.textContent = `${this.uniqueCrossed[dir]} veh`;
        if (queueCell) queueCell.textContent = `${inViewQueue[dir]} veh`;
        if (signalCell) {
          const sig = signals[dir];
          signalCell.className = `signal-pill ${sig.toLowerCase()} cell-signal`;
          signalCell.textContent = sig;
        }
        if (timeCell) {
          const dirPhase = (dir === 'north' || dir === 'south') ? 'PHASE_A_NS' : 'PHASE_B_EW';
          const alloc = this.allocatedGreen[dirPhase] || (dirPhase === 'PHASE_A_NS' ? 35 : 25);
          if (signals[dir] === 'GREEN') {
            timeCell.textContent = `${this.phaseTimeRemaining}s / ${alloc}s`;
          } else if (signals[dir] === 'YELLOW') {
            timeCell.textContent = `${this.transitionTimeRemaining}s / ${alloc}s`;
          } else {
            timeCell.textContent = `0s / ${alloc}s`;
          }
        }
      }
    });

    // Active Phase Name
    let phaseName = 'Phase A: North-South Flow';
    if (this.isEmergency) {
      phaseName = 'EMERGENCY PREEMPTION: Priority Corridor';
    } else if (this.currentPhase === 'PEDESTRIAN') {
      phaseName = 'Pedestrian Crosswalk Phase';
    } else if (this.currentPhase === 'PHASE_B_EW') {
      phaseName = 'Phase B: East-West Flow';
    }
    if (this.transitionState === 'YELLOW') {
      phaseName += ' (Yellow)';
    }

    document.querySelectorAll('.val-active-phase-name').forEach(el => {
      el.textContent = phaseName;
    });

    // Countdown
    const countdownSec = this.transitionState === 'YELLOW' ? this.transitionTimeRemaining : this.phaseTimeRemaining;
    document.querySelectorAll('.val-green-countdown').forEach(el => {
      if (el.closest('.junction-overlay-phase') || el.closest('button')) {
        el.textContent = `${countdownSec}s`;
      } else {
        el.textContent = `${countdownSec}s remaining`;
      }
    });

    // Progress Fill
    const progressFill = document.getElementById('phaseProgressFill');
    if (progressFill && progressFill.style) {
      const totalSec = this.transitionState === 'YELLOW'
        ? this.yellowDuration
        : (this.allocatedGreen[this.currentPhase] || 35);
      const pct = Math.max(0, Math.min(100, Math.round((countdownSec / totalSec) * 100)));
      progressFill.style.width = `${pct}%`;
    }
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw Roads (Cross layout)
    ctx.fillStyle = '#1e293b';
    // Vertical road
    ctx.fillRect(this.cx - this.roadWidth / 2, 0, this.roadWidth, this.canvas.height);
    // Horizontal road
    ctx.fillRect(0, this.cy - this.roadWidth / 2, this.canvas.width, this.roadWidth);

    // Road markings (lane dividers)
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.setLineDash([12, 12]);

    // Vertical lane divider
    ctx.beginPath();
    ctx.moveTo(this.cx, 0);
    ctx.lineTo(this.cx, this.cy - this.roadWidth / 2);
    ctx.moveTo(this.cx, this.cy + this.roadWidth / 2);
    ctx.lineTo(this.cx, this.canvas.height);
    ctx.stroke();

    // Horizontal lane divider
    ctx.beginPath();
    ctx.moveTo(0, this.cy);
    ctx.lineTo(this.cx - this.roadWidth / 2, this.cy);
    ctx.moveTo(this.cx + this.roadWidth / 2, this.cy);
    ctx.lineTo(this.canvas.width, this.cy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Stop Lines (Solid White)
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    // North stop
    ctx.beginPath();
    ctx.moveTo(this.cx - this.roadWidth / 2, this.cy - this.roadWidth / 2 - 10);
    ctx.lineTo(this.cx, this.cy - this.roadWidth / 2 - 10);
    ctx.stroke();
    // South stop
    ctx.beginPath();
    ctx.moveTo(this.cx, this.cy + this.roadWidth / 2 + 10);
    ctx.lineTo(this.cx + this.roadWidth / 2, this.cy + this.roadWidth / 2 + 10);
    ctx.stroke();
    // West stop
    ctx.beginPath();
    ctx.moveTo(this.cx - this.roadWidth / 2 - 10, this.cy);
    ctx.lineTo(this.cx - this.roadWidth / 2 - 10, this.cy + this.roadWidth / 2);
    ctx.stroke();
    // East stop
    ctx.beginPath();
    ctx.moveTo(this.cx + this.roadWidth / 2 + 10, this.cy - this.roadWidth / 2);
    ctx.lineTo(this.cx + this.roadWidth / 2 + 10, this.cy);
    ctx.stroke();

    // Draw Traffic Lights
    const signals = this.getCurrentSignals();
    this.drawTrafficSignal(this.cx - this.roadWidth / 2 - 25, this.cy - this.roadWidth / 2 - 25, signals.north); // North
    this.drawTrafficSignal(this.cx + this.roadWidth / 2 + 10, this.cy + this.roadWidth / 2 + 10, signals.south); // South
    this.drawTrafficSignal(this.cx - this.roadWidth / 2 - 25, this.cy + this.roadWidth / 2 + 10, signals.west); // West
    this.drawTrafficSignal(this.cx + this.roadWidth / 2 + 10, this.cy - this.roadWidth / 2 - 25, signals.east); // East

    // Draw Vehicles
    for (const v of this.vehicles) {
      ctx.save();
      ctx.translate(v.x, v.y);
      if (v.direction === 'north') ctx.rotate(Math.PI);
      else if (v.direction === 'east') ctx.rotate(-Math.PI / 2);
      else if (v.direction === 'west') ctx.rotate(Math.PI / 2);

      // Vehicle body
      ctx.fillStyle = v.color;
      ctx.shadowColor = 'rgba(0,0,0,0.4)';
      ctx.shadowBlur = 6;
      ctx.fillRect(-v.width / 2, -v.length / 2, v.width, v.length);

      // Headlights / taillights
      ctx.fillStyle = '#fef08a';
      ctx.fillRect(-v.width / 2 + 2, v.length / 2 - 2, 4, 2);
      ctx.fillRect(v.width / 2 - 6, v.length / 2 - 2, 4, 2);

      // Ambulance flashing siren
      if (v.isEmergency) {
        ctx.fillStyle = Math.floor(Date.now() / 200) % 2 === 0 ? '#ef4444' : '#3b82f6';
        ctx.beginPath();
        ctx.arc(0, 0, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
    }
  }

  drawTrafficSignal(x, y, signalState) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = '#0f172a';
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, 16, 38);
    ctx.strokeRect(x, y, 16, 38);

    const isRed = signalState === 'RED' || signalState === false;
    const isYellow = signalState === 'YELLOW';
    const isGreen = signalState === 'GREEN' || signalState === true;

    // Red Bulb (top)
    ctx.beginPath();
    ctx.arc(x + 8, y + 7, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = isRed ? '#ef4444' : '#450a0a';
    if (isRed) {
      ctx.shadowColor = '#ef4444';
      ctx.shadowBlur = 8;
    } else {
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    }
    ctx.fill();

    // Yellow Bulb (middle)
    ctx.beginPath();
    ctx.arc(x + 8, y + 19, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = isYellow ? '#f59e0b' : '#451a03';
    if (isYellow) {
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 8;
    } else {
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    }
    ctx.fill();

    // Green Bulb (bottom)
    ctx.beginPath();
    ctx.arc(x + 8, y + 31, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = isGreen ? '#10b981' : '#022c22';
    if (isGreen) {
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 8;
    } else {
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    }
    ctx.fill();

    ctx.restore();
  }
}

// Camera Feed YOLO Overlay Simulator
class CameraStreamSimulator {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.showBoundingBoxes = true;
    this.objects = [
      { label: "Car 94%", x: 40, y: 70, w: 90, h: 60, color: "#10b981", vx: 0.8 },
      { label: "Bus 91%", x: 180, y: 40, w: 140, h: 90, color: "#3b82f6", vx: 0.5 },
      { label: "Motorcycle 88%", x: 110, y: 140, w: 40, h: 40, color: "#f59e0b", vx: 1.2 },
      { label: "Car 92%", x: 260, y: 120, w: 80, h: 55, color: "#10b981", vx: 0.9 },
    ];
    this.loop();
  }

  loop() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Simulated asphalt street background
    ctx.fillStyle = "#334155";
    ctx.fillRect(0, 0, w, h);

    // Road lanes in perspective
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.moveTo(w * 0.35, 0); ctx.lineTo(w * 0.2, h);
    ctx.moveTo(w * 0.65, 0); ctx.lineTo(w * 0.8, h);
    ctx.stroke();
    ctx.setLineDash([]);

    // Vehicles & Bounding boxes
    for (const obj of this.objects) {
      obj.x += obj.vx;
      if (obj.x > w + 20) obj.x = -obj.w - 10;

      // Draw vehicle silhouette
      ctx.fillStyle = "#1e293b";
      ctx.fillRect(obj.x, obj.y, obj.w, obj.h);

      if (this.showBoundingBoxes) {
        ctx.strokeStyle = obj.color;
        ctx.lineWidth = 2;
        ctx.strokeRect(obj.x, obj.y, obj.w, obj.h);

        // Tag label
        ctx.fillStyle = obj.color;
        ctx.fillRect(obj.x, obj.y - 18, ctx.measureText(obj.label).width + 12, 18);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px monospace";
        ctx.fillText(obj.label, obj.x + 4, obj.y - 5);
      }
    }

    requestAnimationFrame(() => this.loop());
  }

  toggleBoxes() {
    this.showBoundingBoxes = !this.showBoundingBoxes;
  }
}
