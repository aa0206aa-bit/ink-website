// fluid.js — 序序流體引擎
// Stable fluids on GPU（Three.js 0.184.0 / WebGL2 / ShaderMaterial GLSL1 語法）
// 對外介面：new InkSimulation(canvas, opts?) / setColor / setMode / setPaper / setLineWidth / clear / splatAt / stir / snapshot / pause / resume / destroy
// 減法混色：dye field 存 absorption（吸收）向量，顯示 paper * exp(-absorption)（Beer-Lambert）
// clear() 為洗い流す式漸淡（約 1.5 秒），非瞬間清空

import * as THREE from './three.module.min.js';

// ===== 可調參數（視覺調校都在這裡）=====
const CONFIG = {
  SIM_RESOLUTION: 144,            // 速度／壓力場短邊格數
  DYE_RESOLUTION_DESKTOP: 1024,   // 染料場短邊（桌機）
  DYE_RESOLUTION_MOBILE: 768,     // 染料場短邊（手機；512 在滿版塗抹時塊狀感明顯）
  PRESSURE_ITERATIONS: 22,        // Jacobi 迭代次數
  CURL_STRENGTH: 24,              // 渦度增強（墨紋捲曲感）
  VELOCITY_DISSIPATION: 0.28,     // 速度消散（越大流動停得越快）
  DYE_DISSIPATION: 0.06,          // 染料消散（接近 0＝墨留在紙上）
  DROP_RADIUS: 0.0035,            // 滴墨染料半徑（uv² 高斯）
  DROP_PULSE: 55,                 // 滴墨徑向速度脈衝強度
  DROP_MOVE_GAP: 0.06,            // 拖曳連滴的最小間距（uv）
  LINE_WIDTHS: { thin: 0.00015, medium: 0.0005, thick: 0.002 }, // 線條墨半徑三檔（uv² 高斯；視覺線寬 ∝ √值）
  LINE_GAP: 0.006,                // 線條下墨步距（密→連續線）
  BLOW_FORCE: 4500,               // 吹墨推力（Task 3 使用）
  BLOW_RADIUS: 0.0007,            // 吹墨作用半徑（細→觸鬚）（Task 3）
  TILT_FORCE: 230,                // 傾斜全域力（Task 3）
  TILT_DECAY: 0.97,               // 鬆手後每幀衰減（Task 3）
  PAPER_COLOR: [0.937, 0.918, 0.878], // 和紙米色 #efeae0（顯示用紙底）
  PAPER_DARK: [0.09, 0.086, 0.102],   // 深紙炭黑 #17161a（發光墨模式）
  INK_STRENGTH_DARK: 0.85,            // 深紙發光墨強度（過高會過曝）
  WASH_FRAMES: 90,                // 洗い流す持續幀數（~1.5s @60fps）
  WASH_DECAY: 0.94,               // 洗墨時每幀 dye 乘法衰減
  ABSORPTION_EPS: 0.012,          // 轉 absorption 時色值下限（防 log(0)）
  INK_STRENGTH: 2.2,              // 一滴墨的濃度係數（sRGB absorption 偏弱，以此補償）
  WHITE_ABSORPTION: -1.0,         // 雲白（留白墨）負吸收強度（有效值 = 此值 × INK_STRENGTH）
  INK_SATURATION: 0.35,           // 紙張吸墨飽和係數（濃度 3 時新墨只染上 ~35%）
};

// ===== Shaders =====
// ShaderMaterial 已自動宣告 position / uv attribute 與 precision，勿重複宣告

const VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// 高斯 splat：對 uTarget 疊加。uRadial=1 時 uValue.x 作徑向脈衝強度（滴墨暈開）
const SPLAT_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uTarget;
uniform float uAspect;
uniform vec3 uValue;
uniform vec2 uPoint;
uniform float uRadius;
uniform float uRadial;
uniform float uSaturation;
void main() {
  vec2 p = vUv - uPoint;
  p.x *= uAspect;
  float fall = exp(-dot(p, p) / uRadius);
  vec3 base = texture2D(uTarget, vUv).xyz;
  vec3 add = uValue;
  if (uRadial > 0.5) {
    vec2 dir = (length(p) > 0.0001) ? normalize(p) : vec2(0.0);
    add = vec3(dir * uValue.x, 0.0);
  }
  // 紙張吸墨飽和：既有濃度越高、新墨越難染上（防滿版疊墨成死黑硬塊）
  // 染料 splat 時 uSaturation > 0；速度 splat 恆為 0（exp(0)=1 不影響）
  add *= exp(-abs(base) * uSaturation);
  gl_FragColor = vec4(base + add * fall, 1.0);
}
`;

// 半拉格朗日平流：座標回溯採樣＋消散
const ADVECTION_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 uTexelSize;
uniform float uDt;
uniform float uDissipation;
void main() {
  vec2 coord = vUv - uDt * texture2D(uVelocity, vUv).xy * uTexelSize;
  vec4 result = texture2D(uSource, coord);
  float decay = 1.0 + uDissipation * uDt;
  gl_FragColor = result / decay;
}
`;

const DIVERGENCE_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform vec2 uTexelSize;
void main() {
  float L = texture2D(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).x;
  float R = texture2D(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).x;
  float B = texture2D(uVelocity, vUv - vec2(0.0, uTexelSize.y)).y;
  float T = texture2D(uVelocity, vUv + vec2(0.0, uTexelSize.y)).y;
  vec2 C = texture2D(uVelocity, vUv).xy;
  if (vUv.x - uTexelSize.x < 0.0) { L = -C.x; }
  if (vUv.x + uTexelSize.x > 1.0) { R = -C.x; }
  if (vUv.y - uTexelSize.y < 0.0) { B = -C.y; }
  if (vUv.y + uTexelSize.y > 1.0) { T = -C.y; }
  float div = 0.5 * (R - L + T - B);
  gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
}
`;

// Jacobi 壓力迭代
const PRESSURE_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform vec2 uTexelSize;
void main() {
  float L = texture2D(uPressure, vUv - vec2(uTexelSize.x, 0.0)).x;
  float R = texture2D(uPressure, vUv + vec2(uTexelSize.x, 0.0)).x;
  float B = texture2D(uPressure, vUv - vec2(0.0, uTexelSize.y)).x;
  float T = texture2D(uPressure, vUv + vec2(0.0, uTexelSize.y)).x;
  float div = texture2D(uDivergence, vUv).x;
  float p = (L + R + B + T - div) * 0.25;
  gl_FragColor = vec4(p, 0.0, 0.0, 1.0);
}
`;

const GRADIENT_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform vec2 uTexelSize;
void main() {
  float L = texture2D(uPressure, vUv - vec2(uTexelSize.x, 0.0)).x;
  float R = texture2D(uPressure, vUv + vec2(uTexelSize.x, 0.0)).x;
  float B = texture2D(uPressure, vUv - vec2(0.0, uTexelSize.y)).x;
  float T = texture2D(uPressure, vUv + vec2(0.0, uTexelSize.y)).x;
  vec2 vel = texture2D(uVelocity, vUv).xy;
  vel -= 0.5 * vec2(R - L, T - B);
  gl_FragColor = vec4(vel, 0.0, 1.0);
}
`;

const CURL_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform vec2 uTexelSize;
void main() {
  float L = texture2D(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).y;
  float R = texture2D(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).y;
  float B = texture2D(uVelocity, vUv - vec2(0.0, uTexelSize.y)).x;
  float T = texture2D(uVelocity, vUv + vec2(0.0, uTexelSize.y)).x;
  float curl = 0.5 * ((R - L) - (T - B));
  gl_FragColor = vec4(curl, 0.0, 0.0, 1.0);
}
`;

// 渦度增強：放大既有旋轉，產生墨紋捲曲
const VORTICITY_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform vec2 uTexelSize;
uniform float uCurlStrength;
uniform float uDt;
void main() {
  float L = texture2D(uCurl, vUv - vec2(uTexelSize.x, 0.0)).x;
  float R = texture2D(uCurl, vUv + vec2(uTexelSize.x, 0.0)).x;
  float B = texture2D(uCurl, vUv - vec2(0.0, uTexelSize.y)).x;
  float T = texture2D(uCurl, vUv + vec2(0.0, uTexelSize.y)).x;
  float C = texture2D(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= (length(force) + 0.0001);
  force *= uCurlStrength * C;
  force.y *= -1.0;
  vec2 vel = texture2D(uVelocity, vUv).xy;
  vel += force * uDt;
  vel = clamp(vel, vec2(-1000.0), vec2(1000.0));
  gl_FragColor = vec4(vel, 0.0, 1.0);
}
`;

// 乘法衰減（壓力場初始猜測）
const CLEAR_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float uValue;
void main() {
  gl_FragColor = uValue * texture2D(uTexture, vUv);
}
`;

// 全域外力（傾斜流動）：對整片速度場加同向力
const FORCE_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform vec2 uForce;
uniform float uDt;
void main() {
  vec2 vel = texture2D(uVelocity, vUv).xy + uForce * uDt;
  gl_FragColor = vec4(vel, 0.0, 1.0);
}
`;

// 染料場 → 螢幕：淺紙＝減法混色（dye＝absorption，墨沉入紙）；深紙＝發光墨（dye＝光色）
// 兩模式共用和紙纖維＋vignette＋dither
const DISPLAY_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uDye;
uniform vec3 uPaper;
uniform float uDark;
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
void main() {
  vec3 ink = texture2D(uDye, vUv).rgb;
  vec3 color;
  if (uDark > 0.5) {
    color = uPaper + ink;              // 深紙：發光墨
  } else {
    color = uPaper * exp(-ink);        // 淺紙：Beer-Lambert 沉墨
    color = min(color, uPaper * 1.04); // 負吸收（雲白）上限：白紙上僅微提亮
  }
  // 和紙纖維：高／中／低三頻 noise 微擾，避免平滑塑膠感
  float fiber = hash(vUv * 720.0) * 0.5 + hash(vUv * 190.0) * 0.3 + hash(vUv * 47.0) * 0.2;
  color *= 1.0 + (fiber - 0.5) * 0.045;
  // 淡 vignette：角落約 11%、邊緣中點約 5% 變暗
  vec2 d = vUv - 0.5;
  color *= 1.0 - dot(d, d) * 0.22;
  float n = hash(vUv * 913.0);
  color += (n - 0.5) / 255.0;
  gl_FragColor = vec4(color, 1.0);
}
`;

// ===== 工具 =====

function createDoubleFBO(w, h, opts) {
  let a = new THREE.WebGLRenderTarget(w, h, opts);
  let b = new THREE.WebGLRenderTarget(w, h, opts);
  return {
    get read() { return a; },
    get write() { return b; },
    swap() { [a, b] = [b, a]; },
    dispose() { a.dispose(); b.dispose(); },
  };
}

// ===== 引擎 =====

export class InkSimulation {
  constructor(canvas, opts = {}) {
    const gl = canvas.getContext('webgl2', {
      alpha: false, depth: false, stencil: false,
      antialias: false, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WEBGL_UNSUPPORTED');

    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, context: gl });
    this.renderer.autoClear = false;
    this.renderer.setClearColor(0x000000, 1);

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);

    this.mode = 'drop';
    this.paperMode = 'light';
    this._lineRadius = CONFIG.LINE_WIDTHS.medium;
    this.color = new THREE.Color('#0f6b63');
    this._absorption = new THREE.Vector3();
    this._toInk(this.color, this._absorption);
    this.gravity = new THREE.Vector2(0, 0);
    this.paused = false;
    this._raf = 0;
    this._lastTime = performance.now();
    this._washFrames = 0;
    this._pointer = { down: false, x: 0, y: 0, lastDropX: -10, lastDropY: -10, tiltStartX: 0, tiltStartY: 0 };
    this._w = 0; this._h = 0;

    const isCoarse = matchMedia('(pointer: coarse)').matches;
    this._dyeRes = opts.dyeResolution ||
      (isCoarse ? CONFIG.DYE_RESOLUTION_MOBILE : CONFIG.DYE_RESOLUTION_DESKTOP);

    this._initMaterials();
    this._resize();
    this._bindEvents();
    this._loopBound = () => this._loop();
    this._raf = requestAnimationFrame(this._loopBound);
    console.log('[InkSim] ready');
  }

  // ---- 公開介面 ----

  setColor(hex) {
    this.color.set(hex);
    this._toInk(this.color, this._absorption);
  }

  setMode(mode) {
    if (mode === 'drop' || mode === 'blow' || mode === 'line' || mode === 'tilt') this.mode = mode;
  }

  // 線條粗細：'thin' | 'medium' | 'thick'
  setLineWidth(width) {
    if (CONFIG.LINE_WIDTHS[width]) this._lineRadius = CONFIG.LINE_WIDTHS[width];
  }

  // 紙色切換：light＝和紙米色（減法混色）／dark＝炭黑（發光墨）。
  // 兩模式 dye 語意不同，切換即立即清空重畫（換紙重來）
  setPaper(mode) {
    if ((mode !== 'light' && mode !== 'dark') || mode === this.paperMode) return;
    this.paperMode = mode;
    const p = mode === 'dark' ? CONFIG.PAPER_DARK : CONFIG.PAPER_COLOR;
    this.mats.display.uniforms.uPaper.value.set(p[0], p[1], p[2]);
    this.mats.display.uniforms.uDark.value = mode === 'dark' ? 1 : 0;
    this._toInk(this.color, this._absorption);
    this._clearImmediate();
  }

  // 洗い流す：速度／壓力立即歸零，墨於 _step 中連續衰減漸淡（~1.5s）
  clear() {
    const r = this.renderer;
    const targets = [
      this.velocity.read, this.velocity.write,
      this.pressure.read, this.pressure.write,
    ];
    for (const rt of targets) {
      r.setRenderTarget(rt);
      r.clear(true, false, false);
    }
    r.setRenderTarget(null);
    this.gravity.set(0, 0);
    this._washFrames = CONFIG.WASH_FRAMES;
  }

  // 立即清空（換紙用：舊 dye 在新模式下語意錯誤，不走漸淡）
  _clearImmediate() {
    const r = this.renderer;
    const targets = [
      this.dye.read, this.dye.write,
      this.velocity.read, this.velocity.write,
      this.pressure.read, this.pressure.write,
    ];
    for (const rt of targets) {
      r.setRenderTarget(rt);
      r.clear(true, false, false);
    }
    r.setRenderTarget(null);
    this.gravity.set(0, 0);
    this._washFrames = 0;
  }

  // 程式化滴墨（自動演出／初始動畫用）。u,v 為 0–1 畫布座標（v 向上），hex 省略用當前墨色
  splatAt(u, v, hex) {
    let a = this._absorption;
    if (hex) {
      a = this._toInk(new THREE.Color(hex), new THREE.Vector3());
    }
    const s = this.paperMode === 'dark' ? CONFIG.INK_STRENGTH_DARK : CONFIG.INK_STRENGTH;
    this._splatDye(u, v, [a.x * s, a.y * s, a.z * s], CONFIG.DROP_RADIUS);
    this._splatVelocity(u, v, CONFIG.DROP_PULSE, 0, CONFIG.DROP_RADIUS * 0.9, true);
  }

  // 程式化輕水流（自動演出用）：在 u,v 注入方向性速度
  stir(u, v, fx, fy) {
    this._splatVelocity(u, v, fx, fy, CONFIG.BLOW_RADIUS * 6);
  }

  // 擷取當前畫布為 <canvas>（同步 render＋readPixels，不依賴 preserveDrawingBuffer）
  snapshot() {
    this._render();
    const gl = this.renderer.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const out = document.createElement('canvas');
    out.width = w; out.height = h;
    const ctx = out.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      img.data.set(px.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
    }
    ctx.putImageData(img, 0, 0);
    return out;
  }

  pause() {
    if (this.paused) return;
    this.paused = true;
    cancelAnimationFrame(this._raf);
  }

  resume() {
    if (!this.paused) return;
    this.paused = false;
    this._lastTime = performance.now();
    this._raf = requestAnimationFrame(this._loopBound);
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    this._unbindEvents();
    this._disposeFBOs();
    for (const key of Object.keys(this.mats)) this.mats[key].dispose();
    this.mesh.geometry.dispose();
    this.renderer.dispose();
  }

  // ---- 內部：初始化 ----

  _mat(frag, uniforms) {
    return new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: frag,
      uniforms,
      depthTest: false,
      depthWrite: false,
    });
  }

  _initMaterials() {
    const v2 = () => ({ value: new THREE.Vector2() });
    this.mats = {
      splat: this._mat(SPLAT_FRAG, {
        uTarget: { value: null }, uAspect: { value: 1 },
        uValue: { value: new THREE.Vector3() }, uPoint: { value: new THREE.Vector2() },
        uRadius: { value: 0.001 }, uRadial: { value: 0 }, uSaturation: { value: 0 },
      }),
      advection: this._mat(ADVECTION_FRAG, {
        uVelocity: { value: null }, uSource: { value: null }, uTexelSize: v2(),
        uDt: { value: 0 }, uDissipation: { value: 0 },
      }),
      divergence: this._mat(DIVERGENCE_FRAG, { uVelocity: { value: null }, uTexelSize: v2() }),
      pressure: this._mat(PRESSURE_FRAG, {
        uPressure: { value: null }, uDivergence: { value: null }, uTexelSize: v2(),
      }),
      gradient: this._mat(GRADIENT_FRAG, {
        uPressure: { value: null }, uVelocity: { value: null }, uTexelSize: v2(),
      }),
      curl: this._mat(CURL_FRAG, { uVelocity: { value: null }, uTexelSize: v2() }),
      vorticity: this._mat(VORTICITY_FRAG, {
        uVelocity: { value: null }, uCurl: { value: null }, uTexelSize: v2(),
        uCurlStrength: { value: CONFIG.CURL_STRENGTH }, uDt: { value: 0 },
      }),
      clear: this._mat(CLEAR_FRAG, { uTexture: { value: null }, uValue: { value: 0.8 } }),
      force: this._mat(FORCE_FRAG, {
        uVelocity: { value: null }, uForce: { value: new THREE.Vector2() }, uDt: { value: 0 },
      }),
      display: this._mat(DISPLAY_FRAG, {
        uDye: { value: null },
        uPaper: { value: new THREE.Vector3(...CONFIG.PAPER_COLOR) },
        uDark: { value: 0 },
      }),
    };
  }

  _disposeFBOs() {
    if (this.velocity) this.velocity.dispose();
    if (this.pressure) this.pressure.dispose();
    if (this.dye) this.dye.dispose();
    if (this.divergenceRT) this.divergenceRT.dispose();
    if (this.curlRT) this.curlRT.dispose();
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (w === this._w && h === this._h) return;
    this._w = w; this._h = h;
    this.renderer.setSize(w, h, false);

    const aspect = w / h;
    const simH = CONFIG.SIM_RESOLUTION;
    const simW = Math.round(simH * aspect);
    const dyeH = Math.min(this._dyeRes, h);
    const dyeW = Math.round(dyeH * aspect);

    const rtOpts = {
      type: THREE.HalfFloatType, format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping,
      depthBuffer: false,
    };

    // 注意：resize 會重建紋理、清空畫面內容（保像素密度、捨內容——設計取捨）
    this._disposeFBOs();
    this.velocity = createDoubleFBO(simW, simH, rtOpts);
    this.pressure = createDoubleFBO(simW, simH, rtOpts);
    this.divergenceRT = new THREE.WebGLRenderTarget(simW, simH, rtOpts);
    this.curlRT = new THREE.WebGLRenderTarget(simW, simH, rtOpts);
    this.dye = createDoubleFBO(dyeW, dyeH, rtOpts);
    this._simTexel = new THREE.Vector2(1 / simW, 1 / simH);
    this._aspect = aspect;
  }

  // ---- 內部：事件 ----

  _uv(e) {
    const r = this.canvas.getBoundingClientRect();
    return [
      (e.clientX - r.left) / r.width,
      1 - (e.clientY - r.top) / r.height,
    ];
  }

  _bindEvents() {
    const c = this.canvas;
    this._onDown = (e) => {
      c.setPointerCapture(e.pointerId);
      const [x, y] = this._uv(e);
      this._pointer.down = true;
      this._pointer.x = x; this._pointer.y = y;
      if (this.mode === 'drop') {
        this._drop(x, y);
        this._pointer.lastDropX = x; this._pointer.lastDropY = y;
      } else if (this.mode === 'line') {
        this._lineDot(x, y);
        this._pointer.lastDropX = x; this._pointer.lastDropY = y;
      } else if (this.mode === 'tilt') {
        this._pointer.tiltStartX = x; this._pointer.tiltStartY = y;
      }
    };
    this._onMove = (e) => {
      if (!this._pointer.down) return;
      const [x, y] = this._uv(e);
      const dx = x - this._pointer.x, dy = y - this._pointer.y;
      if (this.mode === 'drop') {
        const gx = x - this._pointer.lastDropX, gy = y - this._pointer.lastDropY;
        if (gx * gx + gy * gy > CONFIG.DROP_MOVE_GAP * CONFIG.DROP_MOVE_GAP) {
          this._drop(x, y);
          this._pointer.lastDropX = x; this._pointer.lastDropY = y;
        }
      } else if (this.mode === 'blow') {
        // 吹墨：沿移動方向注入細窄強推力，不注入染料（推的是既有的墨）
        this._splatVelocity(x, y, dx * CONFIG.BLOW_FORCE, dy * CONFIG.BLOW_FORCE, CONFIG.BLOW_RADIUS);
      } else if (this.mode === 'line') {
        // 線條：沿軌跡以固定步距插值下細墨（快速拖曳不斷線），不加脈衝——墨線只隨水微暈
        const lx = this._pointer.lastDropX, ly = this._pointer.lastDropY;
        const segLen = Math.hypot(x - lx, y - ly);
        const steps = Math.floor(segLen / CONFIG.LINE_GAP);
        for (let i = 1; i <= steps; i++) {
          const t = (i * CONFIG.LINE_GAP) / segLen;
          this._lineDot(lx + (x - lx) * t, ly + (y - ly) * t);
        }
        if (steps > 0) {
          const t = (steps * CONFIG.LINE_GAP) / segLen;
          this._pointer.lastDropX = lx + (x - lx) * t;
          this._pointer.lastDropY = ly + (y - ly) * t;
        }
      } else if (this.mode === 'tilt') {
        // 傾斜：拖曳向量 → 全域力；拖滿半個畫布寬 = 最大力
        const tx = x - this._pointer.tiltStartX, ty = y - this._pointer.tiltStartY;
        const len = Math.hypot(tx, ty);
        if (len > 0.001) {
          const strength = Math.min(len, 0.5) / 0.5;
          this.gravity.set(
            (tx / len) * CONFIG.TILT_FORCE * strength,
            (ty / len) * CONFIG.TILT_FORCE * strength
          );
        }
      }
      this._pointer.x = x; this._pointer.y = y;
    };
    this._onUp = () => { this._pointer.down = false; };

    c.addEventListener('pointerdown', this._onDown);
    c.addEventListener('pointermove', this._onMove);
    c.addEventListener('pointerup', this._onUp);
    c.addEventListener('pointercancel', this._onUp);

    // iOS Safari 對 touch-action: none 不完全可靠——作畫時頁面仍會捲動；
    // 需 non-passive touchmove preventDefault 雙保險（畫布上不捲頁，畫布外正常捲）
    this._onTouchMove = (e) => e.preventDefault();
    c.addEventListener('touchmove', this._onTouchMove, { passive: false });

    // 手機切換 app 等情境會弄丟 GL context；Three.js 會自行重建 GL 狀態，
    // 但紋理內容已失，clear() 讓模擬回到一致的空白狀態
    this._onCtxLost = (e) => { e.preventDefault(); this.pause(); };
    this._onCtxRestored = () => { this.clear(); this.resume(); };
    c.addEventListener('webglcontextlost', this._onCtxLost);
    c.addEventListener('webglcontextrestored', this._onCtxRestored);

    let t = 0;
    this._ro = new ResizeObserver(() => {
      clearTimeout(t);
      t = setTimeout(() => this._resize(), 200);
    });
    this._ro.observe(c);
  }

  _unbindEvents() {
    const c = this.canvas;
    c.removeEventListener('pointerdown', this._onDown);
    c.removeEventListener('pointermove', this._onMove);
    c.removeEventListener('pointerup', this._onUp);
    c.removeEventListener('pointercancel', this._onUp);
    c.removeEventListener('touchmove', this._onTouchMove);
    c.removeEventListener('webglcontextlost', this._onCtxLost);
    c.removeEventListener('webglcontextrestored', this._onCtxRestored);
    this._ro.disconnect();
  }

  // ---- 內部：模擬 ----

  _run(mat, target) {
    this.mesh.material = mat;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.scene, this.camera);
  }

  _splatVelocity(x, y, fx, fy, radius, radial = false) {
    const u = this.mats.splat.uniforms;
    u.uTarget.value = this.velocity.read.texture;
    u.uAspect.value = this._aspect;
    u.uPoint.value.set(x, y);
    u.uValue.value.set(fx, fy, 0);
    u.uRadius.value = radius;
    u.uRadial.value = radial ? 1 : 0;
    u.uSaturation.value = 0; // 速度場不飽和
    this._run(this.mats.splat, this.velocity.write);
    this.velocity.swap();
  }

  _splatDye(x, y, rgb, radius) {
    const u = this.mats.splat.uniforms;
    u.uTarget.value = this.dye.read.texture;
    u.uAspect.value = this._aspect;
    u.uPoint.value.set(x, y);
    u.uValue.value.set(rgb[0], rgb[1], rgb[2]);
    u.uRadius.value = radius;
    u.uRadial.value = 0;
    u.uSaturation.value = CONFIG.INK_SATURATION; // 紙張吸墨飽和
    this._run(this.mats.splat, this.dye.write);
    this.dye.swap();
  }

  // 墨色 → dye 向量，依紙色模式分派：
  // light：absorption（Beer-Lambert：A = -log(c)，深色吸收多）；接近純白給負吸收（漂白／留白墨）
  // dark：直接 sRGB 光色（發光墨）
  // 皆用 sRGB 分量：THREE.Color 內部是 linear，display 輸出端是 sRGB canvas——
  // 以 linear 計算會讓中濃度墨過飽和偏螢光
  _toInk(color, out) {
    const c = color.clone().convertLinearToSRGB();
    if (this.paperMode === 'dark') {
      return out.set(c.r, c.g, c.b);
    }
    if (c.r > 0.9 && c.g > 0.9 && c.b > 0.9) {
      const wa = CONFIG.WHITE_ABSORPTION;
      return out.set(wa, wa, wa);
    }
    const e = CONFIG.ABSORPTION_EPS;
    out.set(
      -Math.log(Math.max(c.r, e)),
      -Math.log(Math.max(c.g, e)),
      -Math.log(Math.max(c.b, e)),
    );
    return out;
  }

  _drop(x, y) {
    this.splatAt(x, y);
  }

  // 線條的單點下墨：細半徑、無速度脈衝（線要穩，只隨水微暈）
  _lineDot(x, y) {
    const a = this._absorption;
    const s = this.paperMode === 'dark' ? CONFIG.INK_STRENGTH_DARK : CONFIG.INK_STRENGTH;
    this._splatDye(x, y, [a.x * s, a.y * s, a.z * s], this._lineRadius);
  }

  _step(dt) {
    const m = this.mats;

    // 洗い流す：dye 連續乘法衰減直到 wash 結束（uValue 用完還原 0.8——壓力初猜共用此材質）
    if (this._washFrames > 0) {
      m.clear.uniforms.uTexture.value = this.dye.read.texture;
      m.clear.uniforms.uValue.value = CONFIG.WASH_DECAY;
      this._run(m.clear, this.dye.write);
      this.dye.swap();
      m.clear.uniforms.uValue.value = 0.8;
      this._washFrames--;
    }

    if (this.gravity.lengthSq() > 0.25) {
      const f = m.force.uniforms;
      f.uVelocity.value = this.velocity.read.texture;
      f.uForce.value.copy(this.gravity);
      f.uDt.value = dt;
      this._run(m.force, this.velocity.write);
      this.velocity.swap();
      if (!this._pointer.down) this.gravity.multiplyScalar(CONFIG.TILT_DECAY);
    } else if (this.gravity.lengthSq() > 0) {
      this.gravity.set(0, 0);
    }

    m.curl.uniforms.uVelocity.value = this.velocity.read.texture;
    m.curl.uniforms.uTexelSize.value.copy(this._simTexel);
    this._run(m.curl, this.curlRT);

    m.vorticity.uniforms.uVelocity.value = this.velocity.read.texture;
    m.vorticity.uniforms.uCurl.value = this.curlRT.texture;
    m.vorticity.uniforms.uTexelSize.value.copy(this._simTexel);
    m.vorticity.uniforms.uDt.value = dt;
    this._run(m.vorticity, this.velocity.write);
    this.velocity.swap();

    m.divergence.uniforms.uVelocity.value = this.velocity.read.texture;
    m.divergence.uniforms.uTexelSize.value.copy(this._simTexel);
    this._run(m.divergence, this.divergenceRT);

    m.clear.uniforms.uTexture.value = this.pressure.read.texture;
    m.clear.uniforms.uValue.value = 0.8;
    this._run(m.clear, this.pressure.write);
    this.pressure.swap();

    m.pressure.uniforms.uDivergence.value = this.divergenceRT.texture;
    m.pressure.uniforms.uTexelSize.value.copy(this._simTexel);
    for (let i = 0; i < CONFIG.PRESSURE_ITERATIONS; i++) {
      m.pressure.uniforms.uPressure.value = this.pressure.read.texture;
      this._run(m.pressure, this.pressure.write);
      this.pressure.swap();
    }

    m.gradient.uniforms.uPressure.value = this.pressure.read.texture;
    m.gradient.uniforms.uVelocity.value = this.velocity.read.texture;
    m.gradient.uniforms.uTexelSize.value.copy(this._simTexel);
    this._run(m.gradient, this.velocity.write);
    this.velocity.swap();

    m.advection.uniforms.uVelocity.value = this.velocity.read.texture;
    m.advection.uniforms.uSource.value = this.velocity.read.texture;
    m.advection.uniforms.uTexelSize.value.copy(this._simTexel);
    m.advection.uniforms.uDt.value = dt;
    m.advection.uniforms.uDissipation.value = CONFIG.VELOCITY_DISSIPATION;
    this._run(m.advection, this.velocity.write);
    this.velocity.swap();

    m.advection.uniforms.uVelocity.value = this.velocity.read.texture;
    m.advection.uniforms.uSource.value = this.dye.read.texture;
    m.advection.uniforms.uDissipation.value = CONFIG.DYE_DISSIPATION;
    this._run(m.advection, this.dye.write);
    this.dye.swap();
  }

  _render() {
    this.mats.display.uniforms.uDye.value = this.dye.read.texture;
    this._run(this.mats.display, null);
  }

  _loop() {
    this._raf = requestAnimationFrame(this._loopBound);
    const now = performance.now();
    let dt = (now - this._lastTime) / 1000;
    this._lastTime = now;
    dt = Math.min(dt, 1 / 30);
    this._step(dt);
    this._render();
  }
}
