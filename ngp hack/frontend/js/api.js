const API = {
  baseUrl: window.location.origin,

  async getStatus() {
    try {
      const res = await fetch(`${this.baseUrl}/api/status`);
      return await res.json();
    } catch (e) {
      console.warn("Using offline status fallback", e);
      return null;
    }
  },

  async getAnalytics() {
    try {
      const res = await fetch(`${this.baseUrl}/api/analytics`);
      return await res.json();
    } catch (e) {
      console.warn("Using offline analytics fallback", e);
      return null;
    }
  },

  async getConfig() {
    try {
      const res = await fetch(`${this.baseUrl}/api/config`);
      return await res.json();
    } catch (e) {
      console.warn("Using offline config fallback", e);
      return null;
    }
  },

  async saveConfig(config) {
    try {
      const res = await fetch(`${this.baseUrl}/api/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config)
      });
      return await res.json();
    } catch (e) {
      console.error("Failed to save config", e);
      return { success: false };
    }
  },

  async setMode(mode) {
    try {
      const res = await fetch(`${this.baseUrl}/api/junction/mode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode })
      });
      return await res.json();
    } catch (e) {
      return { success: false };
    }
  },

  async triggerPhase(phase) {
    try {
      const res = await fetch(`${this.baseUrl}/api/junction/phase-override`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phase })
      });
      return await res.json();
    } catch (e) {
      return { success: false };
    }
  },

  async triggerEmergency(direction = "South") {
    try {
      const res = await fetch(`${this.baseUrl}/api/junction/emergency`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ direction })
      });
      return await res.json();
    } catch (e) {
      return { success: false };
    }
  },

  // ==================== COMPUTER VISION API ====================

  async uploadCVVideo(approach, file) {
    try {
      const formData = new FormData();
      formData.append('approach', approach);
      formData.append('file', file);
      const res = await fetch(`${this.baseUrl}/api/cv/upload`, {
        method: 'POST',
        body: formData
      });
      return await res.json();
    } catch (e) {
      console.error('Failed to upload video', e);
      return { success: false, error: e.message };
    }
  },

  async loadCVDemo(approach, clip = 'highway') {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/demo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach, clip })
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async processCV(approach = 'all', max_frames = 600) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach, max_frames })
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async stopCV(approach = null) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(approach ? { approach } : {})
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async configureCVApproach(approach, source_type = null, selected = null, ip_url = null) {
    try {
      const payload = { approach };
      if (source_type !== null) payload.source_type = source_type;
      if (selected !== null) payload.selected = selected;
      if (ip_url !== null) payload.ip_url = ip_url;
      const res = await fetch(`${this.baseUrl}/api/cv/approach/configure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async startCVWebcam(approach) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/webcam/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach })
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async stopCVWebcam(approach) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/webcam/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach })
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async startCVIpCam(approach, ip_url = null) {
    try {
      const payload = { approach };
      if (ip_url) payload.ip_url = ip_url;
      const res = await fetch(`${this.baseUrl}/api/cv/ipcam/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async stopCVIpCam(approach) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/ipcam/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach })
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async startCVSim(approach) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/sim/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach })
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async stopCVSim(approach) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/sim/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach })
      });
      return await res.json();
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  async getCVStatus() {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/status`);
      return await res.json();
    } catch (e) {
      return null;
    }
  },

  async calibrateCVLine(approach, line_height, direction = 'any') {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/calibrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach, line_height, direction })
      });
      return await res.json();
    } catch (e) {
      return { success: false };
    }
  },

  async resetCV() {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/reset`, {
        method: 'POST'
      });
      return await res.json();
    } catch (e) {
      return { success: false };
    }
  },

  async setCVMode(is_simulation) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_simulation })
      });
      return await res.json();
    } catch (e) {
      return { success: false };
    }
  },

  async simulateCVCounts(counts) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(counts)
      });
      return await res.json();
    } catch (e) {
      return { success: false };
    }
  },

  async getCVExport() {
    try {
      const res = await fetch(`${this.baseUrl}/api/cv/export`);
      return await res.json();
    } catch (e) {
      return null;
    }
  }
};
