// Main Application Controller
document.addEventListener('DOMContentLoaded', async () => {
  let junctionSim = null;
  let junctionSim2 = null;
  let chartsManager = new TrafficChartsManager();
  let currentScreen = 'dashboard';
  let isSoundEnabled = true;

  // Initialize Canvas Simulators
  junctionSim = new JunctionSimulator('junctionCanvas');
  junctionSim2 = new JunctionSimulator('junctionCanvas2');
  window.junctionSim = junctionSim;
  window.junctionSim2 = junctionSim2;

  // Navigation Setup
  const navItems = document.querySelectorAll('.nav-item');
  const screens = document.querySelectorAll('.screen-view');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const target = item.dataset.screen;
      if (!target) return;

      navItems.forEach(n => n.classList.remove('active'));
      item.classList.add('active');

      screens.forEach(s => s.classList.remove('active'));
      const activeScreen = document.getElementById(`screen-${target}`);
      if (activeScreen) {
        activeScreen.classList.add('active');
        currentScreen = target;
        if (target === 'analytics') {
          loadAnalyticsData();
        } else if (target === 'computer-vision') {
          updateCVScreen();
        }
      }
    });
  });

  // Sound Toggle
  const soundBtn = document.getElementById('soundToggleBtn');
  if (soundBtn) {
    soundBtn.addEventListener('click', () => {
      isSoundEnabled = !isSoundEnabled;
      soundBtn.title = isSoundEnabled ? 'Audio alerts enabled' : 'Audio alerts muted';
      soundBtn.style.opacity = isSoundEnabled ? '1' : '0.4';
    });
  }

  // Audio Beep for alerts
  function playAlertBeep() {
    if (!isSoundEnabled) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.3);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.3);
    } catch (e) {}
  }

  // Live Polling & State Update
  async function pollStatus() {
    const data = await API.getStatus();
    if (!data) return;

    const { status, approaches, config, logs, cv_service } = data;

    // Update Top status
    const remainingSec = status.current_green_remaining;
    const totalSec = status.current_green_total || 45;
    const pct = Math.max(0, Math.min(100, Math.round((remainingSec / totalSec) * 100)));

    // Update elements
    if (!junctionSim) {
      document.querySelectorAll('.val-active-phase-name').forEach(el => el.textContent = status.active_phase_name);
      document.querySelectorAll('.val-green-countdown').forEach(el => el.textContent = `${remainingSec}s`);
    }
    document.querySelectorAll('.val-total-cleared').forEach(el => el.textContent = status.total_vehicles_cleared.toLocaleString());
    document.querySelectorAll('.val-flow-rate').forEach(el => el.textContent = `${status.average_flow_rate} veh/min`);

    // Update CV Data Source indicators on Dashboard
    const cvSourcePill = document.getElementById('cvDataSourcePill');
    if (cvSourcePill) {
      const mode = status.cv_source || (cv_service && cv_service.mode) || 'WAITING FOR VIDEO DATA';
      cvSourcePill.textContent = mode;
      cvSourcePill.className = 'source-pill ' + (
        mode.includes('LIVE') ? 'live' :
        mode.includes('PROCESSED') ? 'processed' :
        mode.includes('SIMULATION') ? 'sim' : 'waiting'
      );
    }

    const cvInViewPill = document.getElementById('cvTotalDetectedPill');
    if (cvInViewPill) {
      const inView = cv_service ? cv_service.total_currently_in_view : (status.total_detected_vehicles || 0);
      cvInViewPill.textContent = `${inView} Vehicles In-View`;
    }

    const cvUniquePill = document.getElementById('cvTotalUniquePill');
    if (cvUniquePill) {
      const unique = cv_service ? cv_service.total_unique_counted : 0;
      cvUniquePill.textContent = `${unique} Total Unique Crossed`;
    }

    const cvEngineStatus = document.getElementById('cvEngineStatus');
    if (cvEngineStatus && cv_service) {
      if (cv_service.detector_error) {
        cvEngineStatus.textContent = 'Degraded (YOLO Offline)';
        cvEngineStatus.style.color = '#ef4444';
      } else {
        cvEngineStatus.textContent = 'Online & Synced';
        cvEngineStatus.style.color = '#10b981';
      }
    }

    // Only update SPaT table, active phase countdown, and progress from backend if junctionSim is not active
    if (!junctionSim) {
      const progressFill = document.getElementById('phaseProgressFill');
      if (progressFill) progressFill.style.width = `${pct}%`;

      if (junctionSim2) junctionSim2.setPhase(status.active_phase, status.emergency_preempt === 1);

      approaches.forEach(app => {
        const dirLower = app.direction.toLowerCase();
        const row = document.getElementById(`row-app-${dirLower}`);
        if (row) {
          const uniqueCell = row.querySelector('.cell-unique');
          const queueCell = row.querySelector('.cell-queue');
          const signalCell = row.querySelector('.cell-signal');
          const timeCell = row.querySelector('.cell-time');
          if (uniqueCell) uniqueCell.textContent = `${app.unique_crossed !== undefined ? app.unique_crossed : 0} veh`;
          if (queueCell) queueCell.textContent = `${app.currently_in_frame !== undefined ? app.currently_in_frame : app.queue_count} veh`;
          if (timeCell) timeCell.textContent = `${app.allocated_green}s`;
          if (signalCell) {
            signalCell.className = `signal-pill ${app.current_signal.toLowerCase()} cell-signal`;
            signalCell.textContent = app.current_signal;
          }
        }
      });
    }

    // Update Logs
    const logList = document.getElementById('eventLogList');
    if (logList && logs) {
      logList.innerHTML = logs.map(l => `
        <li class="event-log-item ${l.level.toLowerCase()}">
          <span class="event-time">${l.timestamp}</span>
          <span class="event-msg"><strong>[${l.source}]</strong> ${l.message}</span>
        </li>
      `).join('');
    }
  }

  // Load Analytics Data
  async function loadAnalyticsData() {
    const data = await API.getAnalytics();
    if (!data) return;
    chartsManager.init(data.history);

    // Update Summary KPI Cards
    if (data.summary) {
      const s = data.summary;
      const elWait = document.getElementById('kpi-avg-waiting');
      const elThroughput = document.getElementById('kpi-throughput');
      const elQueue = document.getElementById('kpi-queue-length');
      const elEmissions = document.getElementById('kpi-emissions');

      if (elWait) elWait.textContent = `${s.avg_waiting_time_adaptive}s`;
      if (elThroughput) elThroughput.textContent = s.total_throughput_adaptive.toLocaleString();
      if (elQueue) elQueue.textContent = `${s.avg_queue_length_adaptive} cars`;
      if (elEmissions) elEmissions.textContent = `${s.fuel_saved_liters} L / ${s.co2_saved_kg} kg`;
    }
  }

  // Analytics Metric Tabs
  document.querySelectorAll('.chart-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.chart-tab-btn').forEach(b => b.classList.remove('active', 'btn-primary'));
      btn.classList.add('active', 'btn-primary');
      chartsManager.setMetric(btn.dataset.metric);
    });
  });

  // Action Buttons: Manual Phase Override
  document.querySelectorAll('[data-action="phase-override"]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const phase = btn.dataset.phase;
      if (junctionSim) junctionSim.triggerManualOverride(phase);
      if (junctionSim2) junctionSim2.triggerManualOverride(phase);
      await API.triggerPhase(phase);
      pollStatus();
    });
  });

  // Emergency Preemption Trigger
  const emergencyBtn = document.getElementById('emergencyPreemptBtn');
  if (emergencyBtn) {
    emergencyBtn.addEventListener('click', async () => {
      playAlertBeep();
      if (junctionSim) junctionSim.triggerEmergencyPreempt('south');
      if (junctionSim2) junctionSim2.triggerEmergencyPreempt('south');
      await API.triggerEmergency('South');
      pollStatus();
    });
  }

  // Emergency Preemption Trigger (Screen 2)
  const emergencyBtn2 = document.getElementById('emergencyPreemptBtn2');
  if (emergencyBtn2) {
    emergencyBtn2.addEventListener('click', async () => {
      playAlertBeep();
      if (junctionSim) junctionSim.triggerEmergencyPreempt('south');
      if (junctionSim2) junctionSim2.triggerEmergencyPreempt('south');
      await API.triggerEmergency('South');
      pollStatus();
    });
  }

  // ==================== COMPUTER VISION MODULE MANAGER ====================
  let isCVSimMode = false;
  let isCVProcessing = false;
  const approachesList = ['North', 'South', 'East', 'West'];

  async function updateCVScreen() {
    const cvStatus = await API.getCVStatus();
    if (!cvStatus) return;

    isCVSimMode = cvStatus.is_simulation;
    isCVProcessing = cvStatus.is_processing;

    // Update Global Headers
    const globalModeEl = document.getElementById('cvGlobalMode');
    if (globalModeEl) {
      if (!cvStatus.is_simulation && cvStatus.detector_available === false) {
        globalModeEl.textContent = 'YOLO UNAVAILABLE';
        globalModeEl.title = cvStatus.detector_error || 'Model weights or ultralytics missing';
        globalModeEl.className = 'cv-summary-val highlight mode-failed';
      } else {
        globalModeEl.textContent = cvStatus.mode;
        globalModeEl.title = '';
        globalModeEl.className = `cv-summary-val highlight ${cvStatus.mode.toLowerCase().replace(/\s+/g, '-')}`;
      }
    }

    const uniqueEl = document.getElementById('cvGlobalUnique');
    if (uniqueEl) uniqueEl.textContent = `${cvStatus.total_unique_counted} veh`;

    const inViewEl = document.getElementById('cvGlobalInView');
    if (inViewEl) inViewEl.textContent = `${cvStatus.total_currently_in_view} veh`;

    const simToggleLabel = document.getElementById('cvSimToggleLabel');
    if (simToggleLabel) {
      simToggleLabel.textContent = isCVSimMode ? 'Mode: Simulation Fallback (Active)' : 'Mode: Real Video (Active)';
    }

    // Process All Button Guard
    const processAllBtn = document.getElementById('cvProcessAllBtn');
    if (processAllBtn) {
      if (isCVProcessing) {
        processAllBtn.disabled = true;
        processAllBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="animation: spin 1s linear infinite;"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>
          Processing Concurrently...
        `;
      } else {
        processAllBtn.disabled = false;
        processAllBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          Process All Approaches
        `;
      }
    }

    // Update 4 Approaches
    approachesList.forEach(app => {
      const info = cvStatus.approaches ? cvStatus.approaches[app] : null;
      if (!info) return;

      // Selection Checkbox
      const selectCb = document.getElementById(`cvSelect-${app}`);
      if (selectCb && document.activeElement !== selectCb) {
        selectCb.checked = Boolean(info.selected);
      }

      // Source Dropdown Selector
      const sourceSelect = document.getElementById(`cvSourceSelect-${app}`);
      if (sourceSelect && document.activeElement !== sourceSelect) {
        if (info.source_type) sourceSelect.value = info.source_type;
      }

      // Source Status Pill
      const sourcePill = document.getElementById(`cvSourcePill-${app}`);
      if (sourcePill) {
        sourcePill.textContent = info.source_status || 'UPLOADED VIDEO';
        sourcePill.className = 'cv-source-pill ' + (
          info.source_status === 'LIVE CAMERA' ? 'live-camera' :
          (info.source_status === 'IP CAMERA' ? 'ip-camera' :
          (info.source_status === 'SIMULATED DATA' ? 'simulated-data' : 'uploaded-video'))
        );
      }

      // Badge
      const badge = document.getElementById(`cvBadge-${app}`);
      if (badge) {
        badge.textContent = info.status;
        badge.className = `cv-badge ${info.status.toLowerCase()}`;
        if (info.error_message) {
          badge.title = info.error_message;
        } else {
          badge.title = '';
        }
      }

      // Video / Source Display Name
      const nameEl = document.getElementById(`cvVideoName-${app}`);
      if (nameEl) {
        nameEl.textContent = info.video_name || (info.has_video ? 'Source ready' : 'No source configured');
      }

      // Action Controls Containers & Notice Elements
      const webcamNotice = document.getElementById(`cvWebcamNotice-${app}`);
      const noticeText = document.getElementById(`cvWebcamNoticeText-${app}`);
      const webcamControls = document.getElementById(`cvWebcamControls-${app}`);
      const fileControls = document.getElementById(`cvFileControls-${app}`);
      const ipCamControls = document.getElementById(`cvIpCamControls-${app}`);
      const simControls = document.getElementById(`cvSimControls-${app}`);

      const webcamStartBtn = document.getElementById(`cvWebcamStartBtn-${app}`);
      const webcamStopBtn = document.getElementById(`cvWebcamStopBtn-${app}`);
      const ipCamStartBtn = document.getElementById(`cvIpCamStartBtn-${app}`);
      const ipCamStopBtn = document.getElementById(`cvIpCamStopBtn-${app}`);
      const simStartBtn = document.getElementById(`cvSimStartBtn-${app}`);
      const simStopBtn = document.getElementById(`cvSimStopBtn-${app}`);
      const processSingleBtn = document.getElementById(`cvProcessSingleBtn-${app}`);
      const processStopBtn = document.getElementById(`cvProcessStopBtn-${app}`);
      const ipUrlInput = document.getElementById(`cvIpUrl-${app}`);

      // Hide all action control rows first
      if (webcamControls) webcamControls.style.display = 'none';
      if (fileControls) fileControls.style.display = 'none';
      if (ipCamControls) ipCamControls.style.display = 'none';
      if (simControls) simControls.style.display = 'none';

      const isWebcamSource = info.source_type === 'webcam';
      const isIpSource = info.source_type === 'ip_camera';
      const isSimSource = info.source_type === 'simulated';
      const isLockedByOther = cvStatus.webcam_holder && cvStatus.webcam_holder !== app;

      if (isWebcamSource) {
        if (webcamControls) webcamControls.style.display = 'flex';

        if (info.status === 'PROCESSING' && cvStatus.webcam_holder === app) {
          if (webcamStartBtn) webcamStartBtn.style.display = 'none';
          if (webcamStopBtn) webcamStopBtn.style.display = 'inline-flex';
        } else {
          if (webcamStartBtn) webcamStartBtn.style.display = 'inline-flex';
          if (webcamStopBtn) webcamStopBtn.style.display = 'none';
        }

        // Shared Webcam In-Use Lock Check
        if (isLockedByOther) {
          if (webcamNotice) webcamNotice.style.display = 'flex';
          if (noticeText) noticeText.textContent = `Webcam already in use by ${cvStatus.webcam_holder}`;
          if (webcamStartBtn) {
            webcamStartBtn.disabled = true;
            webcamStartBtn.title = `Webcam already in use by ${cvStatus.webcam_holder}`;
          }
        } else if (info.status === 'FAILED' && info.error_message) {
          if (webcamNotice) webcamNotice.style.display = 'flex';
          if (noticeText) noticeText.textContent = info.error_message;
          if (webcamStartBtn) {
            webcamStartBtn.disabled = false;
            webcamStartBtn.title = '';
          }
        } else {
          if (webcamNotice) webcamNotice.style.display = 'none';
          if (webcamStartBtn) {
            webcamStartBtn.disabled = false;
            webcamStartBtn.title = '';
          }
        }
      } else if (isIpSource) {
        if (ipCamControls) ipCamControls.style.display = 'flex';
        if (ipUrlInput && document.activeElement !== ipUrlInput && info.ip_url !== undefined) {
          ipUrlInput.value = info.ip_url;
        }

        if (info.status === 'PROCESSING') {
          if (ipCamStartBtn) ipCamStartBtn.style.display = 'none';
          if (ipCamStopBtn) ipCamStopBtn.style.display = 'inline-flex';
        } else {
          if (ipCamStartBtn) ipCamStartBtn.style.display = 'inline-flex';
          if (ipCamStopBtn) ipCamStopBtn.style.display = 'none';
        }

        if (info.status === 'FAILED' && info.error_message) {
          if (webcamNotice) webcamNotice.style.display = 'flex';
          if (noticeText) noticeText.textContent = info.error_message;
        } else {
          if (webcamNotice) webcamNotice.style.display = 'none';
        }
      } else if (isSimSource) {
        if (simControls) simControls.style.display = 'flex';

        if (info.status === 'PROCESSING') {
          if (simStartBtn) simStartBtn.style.display = 'none';
          if (simStopBtn) simStopBtn.style.display = 'inline-flex';
        } else {
          if (simStartBtn) simStartBtn.style.display = 'inline-flex';
          if (simStopBtn) simStopBtn.style.display = 'none';
        }

        if (info.status === 'FAILED' && info.error_message) {
          if (webcamNotice) webcamNotice.style.display = 'flex';
          if (noticeText) noticeText.textContent = info.error_message;
        } else {
          if (webcamNotice) webcamNotice.style.display = 'none';
        }
      } else {
        // Uploaded Video or Demo
        if (fileControls) fileControls.style.display = 'flex';

        if (info.status === 'PROCESSING') {
          if (processSingleBtn) processSingleBtn.style.display = 'none';
          if (processStopBtn) processStopBtn.style.display = 'inline-flex';
        } else {
          if (processSingleBtn) processSingleBtn.style.display = 'inline-flex';
          if (processStopBtn) processStopBtn.style.display = 'none';
        }

        if (info.status === 'FAILED' && info.error_message) {
          if (webcamNotice) webcamNotice.style.display = 'flex';
          if (noticeText) noticeText.textContent = info.error_message;
        } else {
          if (webcamNotice) webcamNotice.style.display = 'none';
        }
      }

      // Progress
      const progEl = document.getElementById(`cvProgress-${app}`);
      if (progEl) {
        if (info.source_type === 'webcam' && info.status === 'PROCESSING') {
          progEl.textContent = 'LIVE';
        } else {
          const pct = info.status === 'COMPLETED' ? 100 : Math.round((info.progress || 0) * 100);
          progEl.textContent = `${pct}%`;
        }
      }

      // Metrics
      const uniqNum = document.getElementById(`cvUnique-${app}`);
      if (uniqNum) uniqNum.textContent = info.total_unique;

      const inViewNum = document.getElementById(`cvInView-${app}`);
      if (inViewNum) inViewNum.textContent = info.currently_in_frame;

      // Class breakdowns
      const cars = document.getElementById(`cvCars-${app}`);
      if (cars) cars.textContent = info.cars || (info.breakdown ? info.breakdown.car : 0);

      const bikes = document.getElementById(`cvBikes-${app}`);
      if (bikes) bikes.textContent = info.motorcycles || (info.breakdown ? info.breakdown.motorcycle : 0);

      const buses = document.getElementById(`cvBuses-${app}`);
      if (buses) buses.textContent = info.buses || (info.breakdown ? info.breakdown.bus : 0);

      const trucks = document.getElementById(`cvTrucks-${app}`);
      if (trucks) trucks.textContent = info.trucks || (info.breakdown ? info.breakdown.truck : 0);

      // Single Approach File Button Guard
      const singleBtn = document.querySelector(`[data-action="process-single"][data-approach="${app}"]`);
      if (singleBtn) {
        if (info.status === 'PROCESSING') {
          singleBtn.disabled = true;
          singleBtn.textContent = 'Processing...';
        } else {
          singleBtn.disabled = false;
          singleBtn.textContent = 'Process Approach';
        }
      }

      // Frame Image vs Placeholder
      const frameImg = document.getElementById(`cvFrame-${app}`);
      const placeholder = document.getElementById(`cvPlaceholder-${app}`);
      const placeholderText = document.getElementById(`cvPlaceholderText-${app}`);

      if (frameImg && placeholder) {
        if (!frameImg.dataset.errorBound) {
          frameImg.onerror = () => {
            frameImg.style.display = 'none';
            placeholder.style.display = 'flex';
          };
          frameImg.dataset.errorBound = 'true';
        }

        if (info.status === 'FAILED') {
          frameImg.style.display = 'none';
          placeholder.style.display = 'flex';
          if (placeholderText) {
            placeholderText.textContent = `❌ ${info.error_message || 'Processing failed'}`;
            placeholderText.style.color = '#ef4444';
          }
        } else if (info.status === 'PROCESSING') {
          if (info.has_frame) {
            frameImg.src = `/api/cv/frame/${app}?t=${Date.now()}`;
            frameImg.style.display = 'block';
            placeholder.style.display = 'none';
          } else {
            frameImg.style.display = 'none';
            placeholder.style.display = 'flex';
            if (placeholderText) {
              placeholderText.textContent = '⏳ Initializing live YOLO detection stream...';
              placeholderText.style.color = '#38bdf8';
            }
          }
        } else if (info.has_frame) {
          frameImg.src = `/api/cv/frame/${app}?t=${Date.now()}`;
          frameImg.style.display = 'block';
          placeholder.style.display = 'none';
        } else {
          frameImg.style.display = 'none';
          placeholder.style.display = 'flex';
          if (placeholderText) {
            placeholderText.style.color = 'var(--text-muted)';
            if (isWebcamSource) {
              placeholderText.textContent = "Click 'Start Camera' to begin YOLO live detection";
            } else if (isIpSource) {
              placeholderText.textContent = info.ip_url ? "Click 'Start Camera' to connect" : "Enter RTSP/HTTP URL and click 'Start Camera'";
            } else if (isSimSource) {
              placeholderText.textContent = "Click 'Start Simulation' to generate vehicle flow";
            } else {
              placeholderText.textContent = "Upload a video or click 'Load Demo Clip'";
            }
          }
        }
      }
    });

    // Update Developer Tools Raw JSON
    const jsonCode = document.getElementById('cvJsonCode');
    if (jsonCode) {
      jsonCode.textContent = JSON.stringify(cvStatus, null, 2);
    }
  }

  // Setup CV Event Listeners
  function setupCVHandlers() {
    // 1. Approach Selection Checkboxes
    document.querySelectorAll('.cv-select-checkbox').forEach(cb => {
      cb.addEventListener('change', async (e) => {
        const app = cb.dataset.approach;
        await API.configureCVApproach(app, null, cb.checked);
        updateCVScreen();
        pollStatus();
      });
    });

    // 2. Source Selector Dropdowns
    document.querySelectorAll('.cv-source-select').forEach(sel => {
      sel.addEventListener('change', async (e) => {
        const app = sel.dataset.approach;
        const newSource = sel.value;
        await API.configureCVApproach(app, newSource, null);
        updateCVScreen();
        pollStatus();
      });
    });

    // 3. Webcam Start & Stop Action Buttons
    document.querySelectorAll('[data-action="webcam-start"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        const res = await API.startCVWebcam(app);
        if (res && !res.success) {
          alert(res.error || 'Failed to start webcam');
        }
        updateCVScreen();
        pollStatus();
      });
    });

    document.querySelectorAll('[data-action="webcam-stop"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        await API.stopCVWebcam(app);
        updateCVScreen();
        pollStatus();
      });
    });

    // 3b. IP Camera Action Buttons
    document.querySelectorAll('[data-action="ipcam-start"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        const input = document.getElementById(`cvIpUrl-${app}`);
        const ipUrl = input ? input.value.trim() : '';
        if (!ipUrl) {
          alert('Please enter an IP Camera RTSP or HTTP URL');
          return;
        }
        const res = await API.startCVIpCam(app, ipUrl);
        if (res && !res.success) {
          alert(res.error || 'Failed to start IP Camera');
        }
        updateCVScreen();
        pollStatus();
      });
    });

    document.querySelectorAll('[data-action="ipcam-stop"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        await API.stopCVIpCam(app);
        updateCVScreen();
        pollStatus();
      });
    });

    // 3c. Simulated Data Action Buttons
    document.querySelectorAll('[data-action="sim-start"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        const res = await API.startCVSim(app);
        if (res && !res.success) {
          alert(res.error || 'Failed to start Simulation');
        }
        updateCVScreen();
        pollStatus();
      });
    });

    document.querySelectorAll('[data-action="sim-stop"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        await API.stopCVSim(app);
        updateCVScreen();
        pollStatus();
      });
    });

    // 3d. Single Approach Process Stop
    document.querySelectorAll('[data-action="process-stop"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        await API.stopCV(app);
        updateCVScreen();
        pollStatus();
      });
    });

    // 4. File inputs, IP inputs & line sliders
    approachesList.forEach(app => {
      const ipInput = document.getElementById(`cvIpUrl-${app}`);
      if (ipInput) {
        ipInput.addEventListener('change', async () => {
          const url = ipInput.value.trim();
          await API.configureCVApproach(app, 'ip_camera', null, url);
        });
      }

      const fileInput = document.getElementById(`cvFile-${app}`);
      if (fileInput) {
        fileInput.addEventListener('change', async (e) => {
          const file = e.target.files[0];
          if (!file) return;
          const badge = document.getElementById(`cvBadge-${app}`);
          if (badge) { badge.textContent = 'UPLOADING...'; badge.className = 'cv-badge processing'; }
          const res = await API.uploadCVVideo(app, file);
          if (res && res.success) {
            updateCVScreen();
            pollStatus();
          } else {
            alert('Upload failed: ' + (res.error || 'Unknown error'));
            if (badge) { badge.textContent = 'FAILED'; badge.className = 'cv-badge failed'; }
          }
        });
      }

      // Slider calibration
      const slider = document.getElementById(`cvLineSlider-${app}`);
      const valLabel = document.getElementById(`cvLineVal-${app}`);
      if (slider && valLabel) {
        slider.addEventListener('input', () => {
          valLabel.textContent = `${slider.value}%`;
        });
        slider.addEventListener('change', async () => {
          await API.calibrateCVLine(app, parseInt(slider.value) / 100.0, 'any');
        });
      }
    });

    // 5. Demo clip load buttons
    document.querySelectorAll('[data-action="load-demo"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        const clip = btn.dataset.clip || 'highway';
        const originalText = btn.textContent;
        btn.textContent = 'Loading...';
        const res = await API.loadCVDemo(app, clip);
        btn.textContent = originalText;
        if (res && res.success) {
          updateCVScreen();
          pollStatus();
        } else {
          alert('Failed to load demo: ' + (res.error || 'Unknown error'));
        }
      });
    });

    // 6. Single approach process buttons
    document.querySelectorAll('[data-action="process-single"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const app = btn.dataset.approach;
        await API.processCV(app, 600);
        updateCVScreen();
        pollStatus();
      });
    });

    // 7. Batch Action: Process All (Concurrent Execution for Selected Approaches)
    const processAllBtn = document.getElementById('cvProcessAllBtn');
    if (processAllBtn) {
      processAllBtn.addEventListener('click', async () => {
        if (processAllBtn.disabled) return;
        await API.processCV('all', 600);
        updateCVScreen();
        pollStatus();
      });
    }

    // 8. Batch Action: Stop
    const stopBtn = document.getElementById('cvStopBtn');
    if (stopBtn) {
      stopBtn.addEventListener('click', async () => {
        await API.stopCV();
        updateCVScreen();
        pollStatus();
      });
    }

    // Batch Action: Reset
    const resetBtn = document.getElementById('cvResetBtn');
    if (resetBtn) {
      resetBtn.addEventListener('click', async () => {
        if (confirm('Reset all vehicle counts and tracking states to 0?')) {
          await API.resetCV();
          updateCVScreen();
          pollStatus();
        }
      });
    }

    // Mode Toggle (Simulation vs Real Video)
    const simToggleBtn = document.getElementById('cvSimToggleBtn');
    if (simToggleBtn) {
      simToggleBtn.addEventListener('click', async () => {
        const newMode = !isCVSimMode;
        await API.setCVMode(newMode);
        updateCVScreen();
        pollStatus();
      });
    }

    // Export Controller JSON
    const exportHandler = async () => {
      const data = await API.getCVExport();
      if (!data) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", "vehicle_counts.json");
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    };

    const exportBtn = document.getElementById('cvExportBtn');
    if (exportBtn) exportBtn.addEventListener('click', exportHandler);

    const dlJsonBtn = document.getElementById('cvDownloadJsonBtn');
    if (dlJsonBtn) dlJsonBtn.addEventListener('click', exportHandler);

    // Copy JSON button
    const copyJsonBtn = document.getElementById('cvCopyJsonBtn');
    if (copyJsonBtn) {
      copyJsonBtn.addEventListener('click', () => {
        const jsonCode = document.getElementById('cvJsonCode');
        if (jsonCode) {
          navigator.clipboard.writeText(jsonCode.textContent).then(() => {
            const original = copyJsonBtn.textContent;
            copyJsonBtn.textContent = 'Copied!';
            setTimeout(() => { copyJsonBtn.textContent = original; }, 2000);
          });
        }
      });
    }

    // Developer Tools Collapsible Drawer
    const devToggle = document.getElementById('cvDevToolsToggle');
    const devContent = document.getElementById('cvDevContent');
    const devIndicator = document.getElementById('cvDevIndicator');
    if (devToggle && devContent) {
      devToggle.addEventListener('click', () => {
        const isHidden = devContent.style.display === 'none';
        devContent.style.display = isHidden ? 'block' : 'none';
        if (devIndicator) {
          devIndicator.textContent = isHidden ? '▼ Click to Collapse' : '▶ Click to Expand';
        }
      });
    }
  }

  // Initialize CV Handlers
  setupCVHandlers();

  // Initial Poll & Dynamic Interval
  await pollStatus();
  await loadAnalyticsData();
  await updateCVScreen();
  setInterval(pollStatus, 1000);
  setInterval(() => {
    if (currentScreen === 'computer-vision' || isCVProcessing) {
      updateCVScreen();
    }
  }, 600);
});
